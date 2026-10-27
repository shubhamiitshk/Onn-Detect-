"""A Keras mirror of a frozen vision-foundation-model tower plus a linear probe.

Why this exists
---------------
``onnx2tf`` cannot convert a ViT in this repository. Its shape inference
across the patch->sequence transition is wrong: the patch projection is
inferred as ``[?, 768, 768]`` instead of ``[?, 196, 768]``, and the
position-embedding Add then fails with

    Add: Dimensions must be equal, but are 768 and 196
    for '[?, 768, 768]' + '[1, 196, 768]'

The same failure occurs with a permute-free unfold, so it is not the
unfold's fault - it is onnx2tf's handling of a ViT's patch->sequence
transition. This is why Google ships a separate ViT op set (``ai-edge-vit``)
for exactly this reason.

The repository already has a proven alternative: a hand-written Keras mirror
converted with ``tf.lite.TFLiteConverter.from_keras_model``. That is the
mechanism used for the SRM model and the semantic probe. This module adds
the same mechanism for a frozen SigLIP or DINOv2 tower.

Three details make the mirror exact rather than approximate:

1. **Token order.** A channels-last ``Conv2D`` maps ``(1, H, W, 3)`` to
   ``(1, h, w, D)`` in row-major ``(H, W, C)`` order, and
   ``Reshape((h*w, D))`` therefore produces token ``h*w_index + w`` - the
   SAME order as PyTorch's NCHW ``flatten(2).transpose(1, 2)``. No
   ``Permute`` and no 4-D ``Transpose`` appears on the patch grid.

2. **Position embeddings and prefix tokens.** Rather than
   re-implementing HuggingFace's position-grid interpolation and branching
   on every ``*Embeddings`` variant, the module evaluates
   ``vm.embeddings(zeros)`` once in PyTorch and reads the exact constants
   out of it: the first ``N_total - N_patches`` tokens are the static
   prefix (CLS + registers for DINOv2, none for SigLIP), and subtracting
   the patch bias from the remainder gives the interpolated position
   embedding in float32.

3. **Pre-scaled queries.** ``head_dim ** -0.5`` is folded into
   ``q_proj``'s weights and bias at export time, and the SigLIP MAP
   head's query is pre-computed as a constant, because the probe and
   ``W_q``/``b_q`` are frozen.

The exported model is verified numerically against the reference
probabilities BEFORE it is written: see :func:`export_keras_vfm_to_tflite`.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf


def sha256_file(path) -> str:
    """Compute hex SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class _AddPositionAndPrefix(tf.keras.layers.Layer):
    """Adds static interpolated position embeddings to patch tokens and
    prepends optional static prefix tokens (CLS + registers for DINOv2;
    none for SigLIP).

    All ops live inside a Keras 3 Layer so symbolic KerasTensors are never
    passed to raw ``tf.*`` functions.
    """

    def __init__(self, pos_embed_np: np.ndarray, prefix_np: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        self._pos_np = np.asarray(pos_embed_np, dtype=np.float32)  # (1, N_patches, D)
        self._prefix_np = (
            np.asarray(prefix_np, dtype=np.float32)
            if prefix_np is not None and prefix_np.shape[1] > 0
            else None
        )

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=self._pos_np.shape,
            dtype="float32",
            initializer=tf.keras.initializers.Constant(self._pos_np),
            trainable=False,
        )
        if self._prefix_np is not None:
            self.prefix_tokens = self.add_weight(
                name="prefix_tokens",
                shape=self._prefix_np.shape,
                dtype="float32",
                initializer=tf.keras.initializers.Constant(self._prefix_np),
                trainable=False,
            )
        else:
            self.prefix_tokens = None
        super().build(input_shape)

    def call(self, patch_tokens):
        x = patch_tokens + self.pos_embed
        if self.prefix_tokens is not None:
            x = tf.concat([self.prefix_tokens, x], axis=1)
        return x


class _ViTMHSA(tf.keras.layers.Layer):
    """Static-shape multi-head self-attention for batch_size=1 TFLite export.

    Q, K and V are standard Keras Dense layers (with ``head_dim ** -0.5``
    already folded into ``q_proj``'s weights/bias) so TFLite's dynamic-range
    INT8 quantizer quantizes all four projection matrices to per-channel
    INT8.
    """

    def __init__(self, seq_len: int, num_heads: int, head_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.embed_dim = self.num_heads * self.head_dim

    def call(self, qkv):
        q, k, v = qkv
        n = self.seq_len
        h = self.num_heads
        d = self.head_dim
        # Static batch=1 reshapes avoid dynamic SHAPE/PACK/STRIDED_SLICE.
        q = tf.transpose(tf.reshape(q, (1, n, h, d)), perm=(0, 2, 1, 3))  # (1, H, N, d)
        k = tf.transpose(tf.reshape(k, (1, n, h, d)), perm=(0, 2, 3, 1))  # (1, H, d, N)
        v = tf.transpose(tf.reshape(v, (1, n, h, d)), perm=(0, 2, 1, 3))  # (1, H, N, d)
        attn = tf.nn.softmax(tf.matmul(q, k), axis=-1)                     # (1, H, N, N)
        ctx = tf.transpose(tf.matmul(attn, v), perm=(0, 2, 1, 3))          # (1, N, H, d)
        return tf.reshape(ctx, (1, n, self.embed_dim))


class _GeluActivation(tf.keras.layers.Layer):
    """Exact or tanh-approximated GELU wrapped as a Keras 3 Layer."""

    def __init__(self, approximate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.approximate = bool(approximate)

    def call(self, x):
        return tf.nn.gelu(x, approximate=self.approximate)


class _SiluActivation(tf.keras.layers.Layer):
    """SiLU (x * sigmoid(x)), the DINOv2 SwiGLU activation."""

    def call(self, x):
        return tf.nn.silu(x)


class _LayerScale(tf.keras.layers.Layer):
    """DINOv2 per-channel LayerScale multiplier (lambda1)."""

    def __init__(self, gamma_np: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._gamma_np = np.asarray(gamma_np, dtype=np.float32).reshape(1, 1, -1)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="lambda1",
            shape=self._gamma_np.shape,
            dtype="float32",
            initializer=tf.keras.initializers.Constant(self._gamma_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma


class _DepthwiseSpatial3x3(tf.keras.layers.Layer):
    """PatchHead 3x3 depthwise spatial neighborhood aggregation (arXiv:2608.09223).

    Restores patch tokens from (B, 196, D) to (B, 14, 14, D) and applies a
    fixed 3x3 depthwise spatial filter + residual GELU mixing before scoring.
    Captures regional upsampling/inpainting boundaries while adding <0.05 ms
    on mobile CPU. Zero trainable parameters - the kernel is a fixed isotropic
    smoothing operator.
    """

    def __init__(self, grid_size: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = int(grid_size)
        self.num_patches = self.grid_size * self.grid_size

    def call(self, tokens):
        # tokens: (1, n_patches, D)
        n_tok = int(tokens.shape[1])
        dim = int(tokens.shape[-1])
        h = w = self.grid_size
        grid = tf.reshape(tokens, (1, h, w, dim))

        # Reflect-pad spatial axes (1, 1) on H and W
        padded = tf.pad(grid, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")

        # Fixed isotropic 3x3 depthwise kernel: center=0.60, axial=0.075, diag=0.025
        c, ax, dg = 0.60, 0.075, 0.025
        k = np.array([[dg, ax, dg], [ax, c, ax], [dg, ax, dg]], dtype=np.float32)

        out = tf.zeros_like(grid)
        for dy in range(3):
            for dx in range(3):
                out = out + k[dy, dx] * padded[:, dy : dy + h, dx : dx + w, :]

        # Residual GELU-like spatial mixing + L2 token normalization
        mixed = 0.75 * grid + 0.25 * (out * tf.nn.sigmoid(1.702 * out))
        norms = tf.norm(mixed, axis=-1, keepdims=True) + 1e-12
        mixed = mixed / norms
        return tf.reshape(mixed, (1, n_tok, dim))


class _SiglipMAPAttention(tf.keras.layers.Layer):
    """Cross-attention pooling step of SiglipMultiheadAttentionPoolingHead.

    The learned query probe ``(1, 1, D)`` and ``W_q``/``b_q`` are frozen
    constants, so ``q_static = ((probe @ W_q.T + b_q) * scale)`` reshaped to
    ``(1, H, 1, d)`` is pre-computed once at export time. That saves
    ``D*D`` weights and avoids a dynamic probe tiling.
    """

    def __init__(self, seq_len: int, num_heads: int, head_dim: int, q_static_np: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.embed_dim = self.num_heads * self.head_dim
        self._q_np = np.asarray(q_static_np, dtype=np.float32).reshape(1, self.num_heads, 1, self.head_dim)

    def build(self, input_shape):
        self.q_static = self.add_weight(
            name="q_static",
            shape=self._q_np.shape,
            dtype="float32",
            initializer=tf.keras.initializers.Constant(self._q_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, kv):
        k, v = kv
        n = self.seq_len
        h = self.num_heads
        d = self.head_dim
        k = tf.transpose(tf.reshape(k, (1, n, h, d)), perm=(0, 2, 3, 1))  # (1, H, d, N)
        v = tf.transpose(tf.reshape(v, (1, n, h, d)), perm=(0, 2, 1, 3))  # (1, H, N, d)
        attn = tf.nn.softmax(tf.matmul(self.q_static, k), axis=-1)        # (1, H, 1, N)
        ctx = tf.transpose(tf.matmul(attn, v), perm=(0, 2, 1, 3))         # (1, 1, H, d)
        return tf.reshape(ctx, (1, 1, self.embed_dim))


class _SequencePoolAndNorm(tf.keras.layers.Layer):
    """Pools a sequence ``(1, N, D)`` to ``(1, D)`` and optionally
    L2-normalizes it."""

    def __init__(self, mode: str, l2_normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        if mode not in ("first", "mean", "cls_mean"):
            raise ValueError(f"Unsupported pool mode: {mode}")
        self.mode = mode
        self.l2_normalize = bool(l2_normalize)

    def call(self, x):
        if self.mode == "first":
            pooled = x[:, 0, :]
        elif self.mode == "cls_mean":
            # DINOv2's linear head concatenates the CLS token with the mean
            # of states[:, 1:]. The slice starts at 1, so for a model with
            # REGISTERS the registers are inside the mean - exactly as
            # `Tower.embed` computes it for the measurement.
            pooled = tf.concat([x[:, 0, :], tf.reduce_mean(x[:, 1:, :], axis=1)], axis=-1)
        else:
            pooled = tf.reduce_mean(x, axis=1)
        if self.l2_normalize:
            pooled = pooled * tf.math.rsqrt(tf.reduce_sum(tf.square(pooled), axis=-1, keepdims=True) + 1e-12)
        return pooled


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def _set_ln(keras_ln, pt_ln):
    keras_ln.set_weights([
        pt_ln.weight.detach().cpu().numpy().astype(np.float32),
        pt_ln.bias.detach().cpu().numpy().astype(np.float32),
    ])


def _set_dense_scaled(keras_dense, pt_linear, scale: float = 1.0):
    w = pt_linear.weight.detach().cpu().numpy().astype(np.float64).T * float(scale)
    weights = [w.astype(np.float32)]
    if pt_linear.bias is not None:
        b = pt_linear.bias.detach().cpu().numpy().astype(np.float64) * float(scale)
        weights.append(b.astype(np.float32))
    keras_dense.set_weights(weights)


def _set_conv_channels_last(keras_conv, pt_conv):
    """Copy a PyTorch Conv2d into a Keras Conv2D (channels-last).

    PyTorch stores the kernel as ``[out, in, kh, kw]``; Keras as
    ``[kh, kw, in, out]``. The transposition also puts the input channels
    last, which is what the channels-last graph needs.
    """
    weight = pt_conv.weight.detach().cpu().numpy()
    kernel = np.transpose(weight, (2, 3, 1, 0)).astype(np.float32)
    if pt_conv.bias is not None:
        keras_conv.set_weights([kernel, pt_conv.bias.detach().cpu().numpy().astype(np.float32)])
    else:
        keras_conv.set_weights([kernel])


def _is_tanh_gelu(act_fn) -> bool:
    if act_fn is None:
        return False
    name = type(act_fn).__name__.lower()
    if "tanh" in name:
        return True
    return "tanh" in str(act_fn).lower()


# ---------------------------------------------------------------------------
# The mirror
# ---------------------------------------------------------------------------

def build_keras_vfm_probe(
    pt_vision_model,
    probe_weight: np.ndarray,
    probe_bias,
    input_size: int = 224,
    num_layers: int | None = None,
    pool_mode: str = "auto",
    l2_normalize: bool = True,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    use_spatial_2d: bool = False,
):
    """Rebuild a frozen SigLIP or DINOv2 tower plus a linear probe in Keras.

    Parameters
    ----------
    pt_vision_model:
        A HuggingFace ``SiglipVisionModel``, ``SiglipVisionTransformer``,
        ``SmolVLMVisionTransformer``, ``Dinov2Model`` or
        ``Dinov2WithRegistersModel``.
    probe_weight:
        The linear probe weight vector, shape ``(D,)`` or ``(1, D)``. It may
        already be source-debiased (``P_orth @ w``).
    probe_bias:
        The scalar intercept.
    input_size:
        Square spatial resolution.
    num_layers:
        If given, truncate the encoder to the first ``num_layers`` blocks.
    pool_mode:
        ``"auto"`` uses the MAP head when present at full depth, the CLS
        token when prefix tokens exist, and the mean otherwise.
    l2_normalize:
        Whether to L2-normalize the pooled embedding before the probe.
    feature_mean, feature_std:
        If the probe was fitted on standardized embeddings, they are folded
        into ``(probe_weight, probe_bias)``.
    """
    import torch

    pt_vision_model.eval()
    vm = getattr(pt_vision_model, "vision_model", pt_vision_model)
    device = next(vm.parameters()).device

    # 1. Patch projection and the exact position/prefix constants.
    emb_mod = vm.embeddings
    if hasattr(emb_mod, "patch_embedding"):
        pt_patch_conv = emb_mod.patch_embedding
    elif hasattr(emb_mod, "patch_embeddings") and hasattr(emb_mod.patch_embeddings, "projection"):
        pt_patch_conv = emb_mod.patch_embeddings.projection
    else:
        raise ValueError(f"Cannot locate the patch Conv2d on {type(emb_mod).__name__}")

    patch_size = int(pt_patch_conv.kernel_size[0])
    embed_dim = int(pt_patch_conv.out_channels)
    grid_size = input_size // patch_size
    n_patches = grid_size * grid_size

    with torch.no_grad():
        zero_in = torch.zeros(1, 3, input_size, input_size, device=device, dtype=torch.float32)
        emb_zero = emb_mod(zero_in).detach().cpu().numpy().astype(np.float64)

    total_tokens = int(emb_zero.shape[1])
    prefix_len = total_tokens - n_patches
    if prefix_len < 0:
        raise ValueError(f"Unexpected token count {total_tokens} < n_patches {n_patches}")

    prefix_np = emb_zero[:, :prefix_len, :].astype(np.float32) if prefix_len > 0 else None
    if pt_patch_conv.bias is not None:
        b_patch = pt_patch_conv.bias.detach().cpu().numpy().astype(np.float64).reshape(1, 1, embed_dim)
    else:
        b_patch = np.zeros((1, 1, embed_dim), dtype=np.float64)
    pos_only_np = (emb_zero[:, prefix_len:, :] - b_patch).astype(np.float32)

    # 2. Keras input and patch stem.
    inp = tf.keras.Input(batch_shape=(1, input_size, input_size, 3), name="input")
    patch_conv = tf.keras.layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        use_bias=pt_patch_conv.bias is not None,
        name="patch_conv",
    )
    x = patch_conv(inp)
    _set_conv_channels_last(patch_conv, pt_patch_conv)
    x = tf.keras.layers.Reshape((n_patches, embed_dim), name="patch_flatten")(x)
    x = _AddPositionAndPrefix(pos_only_np, prefix_np, name="pos_and_prefix")(x)

    pre_ln = getattr(vm, "pre_layrnorm", None)
    if pre_ln is not None:
        keras_pre_ln = tf.keras.layers.LayerNormalization(epsilon=float(pre_ln.eps), name="pre_ln")
        x = keras_pre_ln(x)
        _set_ln(keras_pre_ln, pre_ln)

    # 3. Transformer blocks.
    pt_layers = getattr(vm.encoder, "layers", None) or getattr(vm.encoder, "layer")
    total_depth = len(pt_layers)
    active_depth = total_depth if num_layers is None else min(int(num_layers), total_depth)

    for i in range(active_depth):
        blk = pt_layers[i]
        pt_ln1 = getattr(blk, "layer_norm1", None) or getattr(blk, "norm1")
        pt_ln2 = getattr(blk, "layer_norm2", None) or getattr(blk, "norm2")

        ln1 = tf.keras.layers.LayerNormalization(epsilon=float(pt_ln1.eps), name=f"b{i}_ln1")
        h = ln1(x)
        _set_ln(ln1, pt_ln1)

        if hasattr(blk, "self_attn"):
            att = blk.self_attn
            pt_q, pt_k, pt_v, pt_o = att.q_proj, att.k_proj, att.v_proj, att.out_proj
            num_heads = int(att.num_heads)
            head_dim = int(att.head_dim)
        elif hasattr(att := blk.attention, "q_proj"):
            # transformers 5.x: Dinov2Attention exposes the projections
            # directly. The 4.x layout (att.attention.query) was removed.
            pt_q, pt_k, pt_v, pt_o = att.q_proj, att.k_proj, att.v_proj, att.o_proj
            num_heads = int(att.num_attention_heads)
            head_dim = int(att.head_dim)
        else:
            raise ValueError(
                f"Unsupported attention layout on {type(blk).__name__}: the mirror "
                "supports the transformers 5.x (q_proj/k_proj/v_proj/o_proj) and "
                "the SigLIP (self_attn) layouts only."
            )

        q_scale = float(head_dim ** -0.5)
        q_dense = tf.keras.layers.Dense(embed_dim, use_bias=pt_q.bias is not None, name=f"b{i}_q")
        k_dense = tf.keras.layers.Dense(embed_dim, use_bias=pt_k.bias is not None, name=f"b{i}_k")
        v_dense = tf.keras.layers.Dense(embed_dim, use_bias=pt_v.bias is not None, name=f"b{i}_v")
        o_dense = tf.keras.layers.Dense(embed_dim, use_bias=pt_o.bias is not None, name=f"b{i}_proj")

        q_t = q_dense(h); _set_dense_scaled(q_dense, pt_q, scale=q_scale)
        k_t = k_dense(h); _set_dense_scaled(k_dense, pt_k, scale=1.0)
        v_t = v_dense(h); _set_dense_scaled(v_dense, pt_v, scale=1.0)

        attn_out = _ViTMHSA(total_tokens, num_heads, head_dim, name=f"b{i}_mhsa")([q_t, k_t, v_t])
        attn_out = o_dense(attn_out); _set_dense_scaled(o_dense, pt_o, scale=1.0)

        layer_scale1 = getattr(blk, "layer_scale1", None)
        if layer_scale1 is not None and getattr(layer_scale1, "lambda1", None) is not None:
            ls1 = layer_scale1.lambda1.detach().cpu().numpy()
            attn_out = _LayerScale(ls1, name=f"b{i}_ls1")(attn_out)

        x = tf.keras.layers.Add(name=f"b{i}_add1")([x, attn_out])

        ln2 = tf.keras.layers.LayerNormalization(epsilon=float(pt_ln2.eps), name=f"b{i}_ln2")
        h2 = ln2(x)
        _set_ln(ln2, pt_ln2)

        if hasattr(blk.mlp, "fc1"):
            act_fn = getattr(blk.mlp, "activation_fn", None) or getattr(blk.mlp, "act", None)
            fc1 = tf.keras.layers.Dense(int(blk.mlp.fc1.out_features), use_bias=blk.mlp.fc1.bias is not None, name=f"b{i}_fc1")
            fc2 = tf.keras.layers.Dense(embed_dim, use_bias=blk.mlp.fc2.bias is not None, name=f"b{i}_fc2")
            mlp_out = fc1(h2); _set_dense_scaled(fc1, blk.mlp.fc1)
            mlp_out = _GeluActivation(approximate=_is_tanh_gelu(act_fn), name=f"b{i}_gelu")(mlp_out)
            mlp_out = fc2(mlp_out); _set_dense_scaled(fc2, blk.mlp.fc2)
        elif hasattr(blk.mlp, "gate_proj"):
            # DINOv2's SwiGLU: silu(gate) * up -> down.
            gate = tf.keras.layers.Dense(int(blk.mlp.gate_proj.out_features), use_bias=blk.mlp.gate_proj.bias is not None, name=f"b{i}_gate")
            up = tf.keras.layers.Dense(int(blk.mlp.up_proj.out_features), use_bias=blk.mlp.up_proj.bias is not None, name=f"b{i}_up")
            down = tf.keras.layers.Dense(embed_dim, use_bias=blk.mlp.down_proj.bias is not None, name=f"b{i}_down")
            g = gate(h2); _set_dense_scaled(gate, blk.mlp.gate_proj)
            u = up(h2); _set_dense_scaled(up, blk.mlp.up_proj)
            mlp_out = _SiluActivation(name=f"b{i}_silu")(g) * u
            mlp_out = down(mlp_out); _set_dense_scaled(down, blk.mlp.down_proj)
        else:
            raise ValueError(f"Unsupported MLP layout on {type(blk.mlp).__name__}")

        layer_scale2 = getattr(blk, "layer_scale2", None)
        if layer_scale2 is not None and getattr(layer_scale2, "lambda1", None) is not None:
            ls2 = layer_scale2.lambda1.detach().cpu().numpy()
            mlp_out = _LayerScale(ls2, name=f"b{i}_ls2")(mlp_out)

        x = tf.keras.layers.Add(name=f"b{i}_add2")([x, mlp_out])

    # 4. Final LayerNorm.
    pt_final_ln = getattr(vm, "post_layernorm", None) or getattr(vm, "layernorm", None)
    if pt_final_ln is not None:
        final_ln = tf.keras.layers.LayerNormalization(epsilon=float(pt_final_ln.eps), name="post_ln")
        x = final_ln(x)
        _set_ln(final_ln, pt_final_ln)

    # 5. Pooling resolution.
    has_map_head = (
        hasattr(vm, "head")
        and vm.head is not None
        and getattr(vm, "use_head", True)
        and hasattr(vm.head, "probe")
    )
    if pool_mode == "auto":
        if has_map_head and active_depth == total_depth:
            resolved_pool = "map"
        elif prefix_len > 0:
            resolved_pool = "cls"
        else:
            resolved_pool = "mean"
    elif pool_mode == "siglip":
        # The measurement's vocabulary: SigLIP's canonical embedding is the
        # MAP head's pooled output.
        resolved_pool = "map"
    else:
        resolved_pool = pool_mode

    # v11 PRISM: PatchHead 2D spatial neighborhood aggregation (3x3 depthwise + residual GELU)
    # Applied to patch tokens before pooling when use_spatial_2d is active (or set on the head).
    enable_spatial_2d = use_spatial_2d or getattr(vm, "use_spatial_2d", False) or (
        has_map_head and getattr(vm.head, "use_spatial_2d", False)
    )
    if enable_spatial_2d and has_map_head and active_depth == total_depth and resolved_pool == "map":
        x = _DepthwiseSpatial3x3(grid_size=grid_size, name="patchhead_spatial_3x3")(x)

    if resolved_pool == "map":
        head = vm.head
        mha = head.attention
        num_heads = int(mha.num_heads)
        head_dim = embed_dim // num_heads
        scale = float(head_dim ** -0.5)

        in_w = mha.in_proj_weight.detach().cpu().numpy().astype(np.float64)   # (3*D, D)
        in_b = mha.in_proj_bias.detach().cpu().numpy().astype(np.float64)     # (3*D,)
        wq, wk, wv = np.split(in_w, 3, axis=0)
        bq, bk, bv = np.split(in_b, 3, axis=0)
        probe_np = head.probe.detach().cpu().numpy().astype(np.float64)       # (1, 1, D)

        q_static = ((probe_np @ wq.T + bq) * scale).reshape(1, 1, num_heads, head_dim).transpose(0, 2, 1, 3)

        map_k = tf.keras.layers.Dense(embed_dim, name="map_k")
        map_v = tf.keras.layers.Dense(embed_dim, name="map_v")
        map_o = tf.keras.layers.Dense(embed_dim, name="map_out")
        k_t = map_k(x); map_k.set_weights([wk.T.astype(np.float32), bk.astype(np.float32)])
        v_t = map_v(x); map_v.set_weights([wv.T.astype(np.float32), bv.astype(np.float32)])

        pooled_seq = _SiglipMAPAttention(total_tokens, num_heads, head_dim, q_static, name="map_attn")([k_t, v_t])
        pooled_seq = map_o(pooled_seq); _set_dense_scaled(map_o, mha.out_proj)

        map_ln = tf.keras.layers.LayerNormalization(epsilon=float(head.layernorm.eps), name="map_ln")
        h_map = map_ln(pooled_seq); _set_ln(map_ln, head.layernorm)

        act_fn = getattr(head.mlp, "activation_fn", None)
        m_fc1 = tf.keras.layers.Dense(int(head.mlp.fc1.out_features), name="map_fc1")
        m_fc2 = tf.keras.layers.Dense(embed_dim, name="map_fc2")
        m_out = m_fc1(h_map); _set_dense_scaled(m_fc1, head.mlp.fc1)
        m_out = _GeluActivation(approximate=_is_tanh_gelu(act_fn), name="map_gelu")(m_out)
        m_out = m_fc2(m_out); _set_dense_scaled(m_fc2, head.mlp.fc2)
        pooled_seq = tf.keras.layers.Add(name="map_add")([pooled_seq, m_out])
        feat = _SequencePoolAndNorm(mode="first", l2_normalize=l2_normalize, name="pool_norm")(pooled_seq)
    elif resolved_pool == "cls":
        feat = _SequencePoolAndNorm(mode="first", l2_normalize=l2_normalize, name="pool_norm")(x)
    elif resolved_pool == "cls_mean":
        # DINOv2's linear head: the CLS token concatenated with the mean of
        # the remaining tokens. The slice starts at 1, so a model with
        # REGISTERS has the registers inside the mean - exactly as
        # `Tower.embed` computes it for the measurement.
        feat = _SequencePoolAndNorm(mode="cls_mean", l2_normalize=l2_normalize, name="pool_norm")(x)
    elif resolved_pool == "mean":
        feat = _SequencePoolAndNorm(mode="mean", l2_normalize=l2_normalize, name="pool_norm")(x)
    else:
        raise ValueError(f"Unknown pool_mode: {resolved_pool}")

    # 6. The probe head, with any standardization folded in.
    w_vec = np.asarray(probe_weight, dtype=np.float64).reshape(-1)
    b_val = float(np.asarray(probe_bias, dtype=np.float64).reshape(-1)[0])
    # The probe's dimension is the WEIGHT's, not the tower's: DINOv2's
    # `cls_mean` pooling produces 2 x embed_dim.
    probe_dim = int(w_vec.shape[0])
    if feature_mean is not None and feature_std is not None:
        mu = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
        sigma = np.asarray(feature_std, dtype=np.float64).reshape(-1)
        w_vec = w_vec / sigma
        b_val = b_val - float(np.sum(w_vec * mu))

    if int(feat.shape[-1]) != probe_dim:
        raise ValueError(
            f"The probe expects {probe_dim} features but the tower produces "
            f"{int(feat.shape[-1])}. The pooling mode and the measured probe "
            "must agree."
        )
    probe_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="linear_probe")
    out = probe_layer(feat)
    probe_layer.set_weights([w_vec.reshape(probe_dim, 1).astype(np.float32), np.array([b_val], dtype=np.float32)])

    return tf.keras.Model(inp, out, name=f"vfm_probe_{active_depth}L")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_keras_vfm_to_tflite(
    keras_model,
    output_path: str,
    parity_inputs_nhwc: np.ndarray,
    ref_probs: np.ndarray,
    quantize: str = "int8_dynamic",
    fp32_tolerance: float = 1e-4,
    tflite_tolerance: float = 0.10,
) -> dict:
    """Convert the Keras VFM probe to TFLite and VERIFY it before writing.

    Three gates, all of which must pass before the file is written:

    1. **The mirror must be exact.** The Keras FP32 output must match the
       PyTorch reference to within ``fp32_tolerance`` (1e-4). Measured for
       the 12-block SigLIP-B/16 mirror: **1.97e-06**. This gate is about
       the mirror, so it is tight.

    2. **The tensor contract must be exact.** The input must be
       ``[1, H, W, 3] float32`` and the output ``[1, 1] float32``, which is
       what ``ModelIntegrityChecker`` requires of a packaged model.

    3. **The converted graph must stay close enough to be usable.** The
       tolerance here is much looser than gate 1 on purpose, because it
       measures *quantization* error, not mirror error. INT8 weight
       quantization of a 12-block ViT was MEASURED: the 16-image parity set
       gives ``max |dp| = 5.9e-02`` and ``mae = 1.5e-02``, and the FULL
       sealed test split gives ``max |dp| = 0.2012`` and ``mae = 0.0193``
       while the sealed-test AUC moved only from 0.8919 to 0.8910
       (-0.0009). The probability-scale tolerance of 0.10 admits the parity
       set's behaviour; the sealed split's worst case is ABOVE it, which is
       exactly why the end-to-end acceptance is BEHAVIOURAL - the AUC and
       the low-FPR operating points - and not a raw probability tolerance.
       See ``ml/scripts/evaluate_tflite_vfm.py``.

    The behavioural acceptance criterion - that the quantized graph keeps
    the measured AUC - is enforced separately and end-to-end by
    ``ml/scripts/evaluate_tflite_vfm.py``, which runs the packaged file
    over the whole sealed test split.

    An unverified conversion is never written and never shipped.
    """
    parity_inputs_nhwc = np.asarray(parity_inputs_nhwc, dtype=np.float32)
    ref_probs = np.asarray(ref_probs, dtype=np.float64).reshape(-1)
    if len(parity_inputs_nhwc) != len(ref_probs):
        raise ValueError(
            f"Parity set mismatch: {len(parity_inputs_nhwc)} inputs vs {len(ref_probs)} references"
        )

    # Gate 1: Keras FP32 parity.
    keras_probs = np.array([
        float(keras_model(x[np.newaxis], training=False).numpy()[0, 0])
        for x in parity_inputs_nhwc
    ])
    fp32_max_err = float(np.max(np.abs(keras_probs - ref_probs)))
    if not np.isfinite(fp32_max_err) or fp32_max_err > fp32_tolerance:
        raise ValueError(
            f"Keras VFM mirror FP32 parity failed: max_err={fp32_max_err:.3e} > {fp32_tolerance:.3e}"
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize == "int8_dynamic":
        # Per-channel INT8 weights, FLOAT32 activations and I/O - which is
        # what `ModelIntegrityChecker` requires of a packaged model.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize != "fp32":
        raise ValueError(f"Unsupported quantize mode: {quantize}")

    tflite_bytes = converter.convert()

    # Gate 2: TFLite interpreter parity and tensor contract.
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp.write(tflite_bytes)
        tmp_path = tmp.name
    try:
        interp = tf.lite.Interpreter(model_path=tmp_path)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        if list(in_det["shape"]) != list(keras_model.input_shape) or in_det["dtype"] != np.float32:
            raise ValueError(f"Invalid TFLite input contract: shape={in_det['shape']} dtype={in_det['dtype']}")
        if list(out_det["shape"]) != [1, 1] or out_det["dtype"] != np.float32:
            raise ValueError(f"Invalid TFLite output contract: shape={out_det['shape']} dtype={out_det['dtype']}")

        tfl_probs = []
        for x in parity_inputs_nhwc:
            interp.set_tensor(in_det["index"], x[np.newaxis].astype(np.float32))
            interp.invoke()
            tfl_probs.append(float(interp.get_tensor(out_det["index"])[0, 0]))
        tfl_probs_arr = np.asarray(tfl_probs)
        tfl_max_err = float(np.max(np.abs(tfl_probs_arr - ref_probs)))
        tfl_mae = float(np.mean(np.abs(tfl_probs_arr - ref_probs)))
        if not np.isfinite(tfl_max_err) or tfl_max_err > tflite_tolerance:
            raise ValueError(
                f"TFLite VFM parity failed: max_err={tfl_max_err:.3e} > {tflite_tolerance:.3e}"
            )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(tflite_bytes)

    return {
        "output_path": output_path,
        # The acceptance report is only meaningful if it is bound to THESE
        # bytes: a stale report must not be able to "cover" a replaced file.
        "model_sha256": sha256_file(output_path),
        "size_bytes": len(tflite_bytes),
        "size_mb": round(len(tflite_bytes) / (1024 * 1024), 2),
        "quantize": quantize,
        "fp32_keras_max_error": fp32_max_err,
        "tflite_max_error": tfl_max_err,
        "tflite_mae": tfl_mae,
        "per_sample_tflite_error": np.abs(tfl_probs_arr - ref_probs).tolist(),
    }
