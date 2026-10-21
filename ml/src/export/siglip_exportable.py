"""An ONNX-exportable SigLIP vision tower for the LiteRT/TFLite path.

Why this exists
---------------
The first attempt to export a frozen SigLIP-B/16 to TFLite failed inside
onnx2tf with::

    LayerNormalization: has 768 elements, but got shape (196,) with 196 elements
    onnx_op_name: .../embeddings/patch_embedding/Conv

The cause is layout handling, not arithmetic. SigLIP's patch projection is
a ``Conv2d(3, 768, kernel_size=16, stride=16)``. onnx2tf rewrites 4-D
convolutions from NCHW to NHWC and then loses track of the channel axis
across the ``Flatten -> Transpose -> Add(position_embedding)`` sequence, so
the first LayerNorm normalises over the 196 patch positions instead of the
768 hidden channels.

The fix is to remove every 4-D convolution from the exported graph. The
patch projection is rewritten as an explicit **patch unfold followed by a
linear map**, which is mathematically identical:

    Conv2d(3, 768, k=16, s=16) applied to x
        == Linear(768, 768) applied to unfold(x) with the conv weight
           reshaped from [768, 3, 16, 16] to [768, 768]

The conv weight is stored as ``[out, in, kh, kw]``, so reshaping it to
``[out, in*kh*kw]`` flattens ``(c, u, v)`` in exactly the order the unfold
produces, which is what makes the two forms bit-identical.

With no 4-D convolution left, onnx2tf never triggers the NCHW->NHWC
rewriting, and the exported graph is a plain stack of MatMul/Softmax/LayerNorm
operations that onnx2tf converts reliably.

The position embedding is frozen to its static ``[1, 196, 768]`` value so
the graph carries no ``Gather`` against a position-index tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExportableSiglipProbe(nn.Module):
    """A SigLIP vision tower plus a linear probe, rewritten for ONNX export.

    The input is **NCHW** ``[B, 3, H, W]`` in ``[0, 1]``. onnx2tf consumes
    NCHW and emits an **NHWC** TFLite, which is the layout every other
    model in this app already uses.

    The output is ``[B, 1]`` - the probability that the image is synthetic.
    """

    def __init__(
        self,
        vision_model: nn.Module,
        probe_weight: torch.Tensor,
        probe_bias: torch.Tensor,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
        pix_mean: torch.Tensor | None = None,
        pix_std: torch.Tensor | None = None,
        unfold_mode: str = "permute",
    ) -> None:
        super().__init__()
        if unfold_mode not in ("permute", "stack"):
            raise ValueError(f"Unknown unfold mode {unfold_mode!r}")
        self.unfold_mode = unfold_mode
        # In transformers 5.x `SiglipVisionModel` IS the vision tower;
        # older versions nest it under `.vision_model`.
        tower = getattr(vision_model, "vision_model", vision_model)
        embeddings = tower.embeddings

        conv = embeddings.patch_embedding
        if conv.weight.ndim != 4:
            raise ValueError(
                "Expected a 4-D patch projection; this wrapper only handles "
                f"the standard Conv2d patch embedding, got {tuple(conv.weight.shape)}"
            )
        out_channels, in_channels, kernel_h, kernel_w = conv.weight.shape
        if kernel_h != kernel_w:
            raise ValueError("Only square patch projections are supported")

        self.patch_size = kernel_h
        self.embed_dim = out_channels
        self.patch_in_dim = in_channels * kernel_h * kernel_w

        # Patch projection as a Linear over unfolded patches.
        self.patch_proj = nn.Linear(self.patch_in_dim, out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            # [out, in, kh, kw] -> [out, in*kh*kw]; the (in, kh, kw) order
            # matches the unfold below exactly.
            self.patch_proj.weight.copy_(conv.weight.reshape(out_channels, -1))
            if conv.bias is not None:
                self.patch_proj.bias.copy_(conv.bias)

        # The position embedding is kept as an Embedding + Gather, exactly as
        # the original tower does it, rather than frozen into a static
        # constant. onnx2tf handles the Gather; a frozen [1, 196, 768]
        # constant is what produces the "768 elements, but got shape (196,)"
        # failure in the first LayerNorm.
        self.position_embedding = nn.Embedding(
            embeddings.position_embedding.num_embeddings,
            embeddings.position_embedding.embedding_dim,
        )
        with torch.no_grad():
            self.position_embedding.weight.copy_(embeddings.position_embedding.weight)
        self.register_buffer("position_ids", embeddings.position_ids.detach().clone())

        self.encoder = tower.encoder
        self.post_layernorm = tower.post_layernorm
        self.head = tower.head

        # The probe head must reproduce the MEASURED function exactly:
        # pool -> L2-normalise -> standardise -> linear -> sigmoid.
        embed_dim = out_channels
        if feat_mean is None:
            feat_mean = torch.zeros(embed_dim)
        if feat_std is None:
            feat_std = torch.ones(embed_dim)
        if pix_mean is None:
            pix_mean = torch.zeros(3)
        if pix_std is None:
            pix_std = torch.ones(3)
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("feat_std", torch.as_tensor(feat_std, dtype=torch.float32).reshape(-1))
        # PIXEL normalisation is the tower's own; it is NOT the feature
        # statistics. Conflating the two is a silent numerical bug.
        self.register_buffer("pix_mean", torch.as_tensor(pix_mean, dtype=torch.float32).reshape(-1))
        self.register_buffer("pix_std", torch.as_tensor(pix_std, dtype=torch.float32).reshape(-1))

        self.probe = nn.Linear(embed_dim, 1)
        with torch.no_grad():
            self.probe.weight.copy_(torch.as_tensor(probe_weight, dtype=torch.float32).reshape(1, embed_dim))
            self.probe.bias.copy_(torch.as_tensor(probe_bias, dtype=torch.float32).reshape(1))

    def patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, N, embed_dim]."""
        batch, channels, height, width = x.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if grid_h * self.patch_size != height or grid_w * self.patch_size != width:
            raise ValueError(
                f"Input {height}x{width} is not a multiple of the patch size {self.patch_size}"
            )
        patches = (
            x.reshape(batch, channels, grid_h, self.patch_size, grid_w, self.patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid_h * grid_w, self.patch_in_dim)
        )
        return self.patch_proj(patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, W, 3] float32 in [0, 1] -> [B, 1] probability.

        The graph input is **NHWC**, which is the layout every model in
        this app already uses and the layout onnx2tf's auto-calibration
        expects (it requires a 4-D input whose last dimension is 3 or 4).
        A NCHW ONNX input makes onnx2tf reject the model before it
        starts, and would also mean a transpose on device.
        """
        if x.shape[-1] != 3:
            raise ValueError(f"Expected NHWC input [B, H, W, 3], got {tuple(x.shape)}")
        rgb = x.permute(0, 3, 1, 2)                       # [B, 3, H, W]
        rgb = (rgb - self.pix_mean.view(1, 3, 1, 1)) / self.pix_std.view(1, 3, 1, 1)
        hidden = self.patch_features(rgb) + self.position_embedding(self.position_ids)
        # The ENCODER is called, not the individual layers. In transformers
        # 5.x a `SiglipEncoderLayer` returns a TENSOR (not a tuple), and
        # the encoder is what wires the attention implementation and the
        # shared kwargs together. Calling the layers directly both changes
        # the semantics and silently slices the batch dimension.
        encoder_out = self.encoder(inputs_embeds=hidden, attention_mask=None)
        hidden = encoder_out.last_hidden_state
        hidden = self.post_layernorm(hidden)
        pooled = self.head(hidden).float()
        # The measured probe operates on L2-NORMALISED features.
        pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-12)
        logits = ((pooled - self.feat_mean) / self.feat_std) @ self.probe.weight.reshape(-1) + self.probe.bias.reshape(-1)
        return torch.sigmoid(logits).reshape(-1, 1)
