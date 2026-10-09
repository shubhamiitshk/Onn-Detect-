"""PyTorch -> Keras -> TFLite conversion for ONN-DETECT models.

ai-edge-torch requires torch_xla, which has no Windows wheels. These two
architectures are small and built entirely from Conv/BN/Hardswish/MFM/Dense,
so we rebuild them in Keras (channels-last) and transfer the trained weights
tensor-by-tensor. Numerical equivalence is verified afterwards by
validate_export.validate_models_real on REAL inputs.

Layout notes for the exported TFLite models (Android contract):
- srm_forensic_cnn_int8.tflite : input (1, 32, 32, 3) float RGB in [0,1]
                                 output (1, 1) P(AI-generated)
- lcnn_audio_int8.tflite       : input (1, T, 40, 1) float LFCC
                                 (T = 401 for 4 s @ 16 kHz) output (1, 1)
"""

import numpy as np
import tensorflow as tf


def _hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _conv_kernel(w: np.ndarray) -> np.ndarray:
    """PyTorch (O, I, kH, kW) -> Keras kernel (kH, kW, I, O)."""
    return np.transpose(w, (2, 3, 1, 0))


def _dw_kernel(w: np.ndarray) -> np.ndarray:
    """PyTorch depthwise (C, 1, kH, kW) -> Keras (kH, kW, C, 1)."""
    return np.transpose(w, (2, 3, 0, 1))


def _set_conv(layer, pt_conv):
    layer.set_weights([_conv_kernel(pt_conv.weight.detach().numpy())]
                      + ([pt_conv.bias.detach().numpy()] if pt_conv.bias is not None else []))


def _set_dw(layer, pt_conv):
    layer.set_weights([_dw_kernel(pt_conv.weight.detach().numpy())])


def _set_bn(layer, pt_bn):
    layer.set_weights([
        pt_bn.weight.detach().numpy(),
        pt_bn.bias.detach().numpy(),
        pt_bn.running_mean.detach().numpy(),
        pt_bn.running_var.detach().numpy(),
    ])


def _set_dense(layer, pt_linear):
    layer.set_weights([pt_linear.weight.detach().numpy().T,
                       pt_linear.bias.detach().numpy()])


def _bn(name):
    # PyTorch BatchNorm2d uses eps=1e-5; Keras defaults to 1e-3.
    return tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name)


class _MFM(tf.keras.layers.Layer):
    """MaxFeatureMap: max over the two halves of the channel/feature axis."""

    def call(self, x):
        c = int(x.shape[-1])
        return tf.maximum(x[..., : c // 2], x[..., c // 2:])


# --------------------------------------------------------------------------
# SRMForensicCNN
# --------------------------------------------------------------------------

def build_keras_srm(pt_model, input_size: int = 32) -> tf.keras.Model:
    """Rebuild SRMForensicCNN channels-last and copy trained weights."""
    pt = pt_model

    inp = tf.keras.Input((input_size, input_size, 3), name="input")
    # SRM fixed filters + tanh (frozen in PyTorch)
    srm_layer = tf.keras.layers.Conv2D(
        3, 5, strides=1, padding="same", use_bias=False,
        trainable=False, name="srm_conv")
    x = srm_layer(inp)
    srm_layer.set_weights([pt.srm.conv.weight.detach().numpy()
                           .transpose(2, 3, 1, 0)])
    x = tf.keras.layers.Activation("tanh", name="srm_tanh")(x)

    # Backbone stem: Conv2d(3,16,3,s2,p1) + BN + Hardswish.
    # PyTorch pads symmetrically (+-1); Keras SAME with stride 2 pads
    # asymmetrically (bottom/right), shifting the sampling grid. Replicate
    # PT exactly: explicit ZeroPadding2D + valid conv.
    x = tf.keras.layers.ZeroPadding2D(1, name="stem_pad")(x)
    c1 = tf.keras.layers.Conv2D(16, 3, strides=2, padding="valid",
                                use_bias=False, name="stem_conv")
    x = c1(x); _set_conv(c1, pt.backbone[0])
    b1 = _bn("stem_bn"); x = b1(x); _set_bn(b1, pt.backbone[1])
    x = tf.keras.layers.Activation(_hard_swish, name="stem_hswish")(x)

    # DepthwiseSeparableBlocks: backbone indices 3..6
    ds_specs = [(16, 32, 1), (32, 64, 2), (64, 128, 2), (128, 256, 2)]
    for i, (cin, cout, stride) in enumerate(ds_specs):
        pt_block = pt.backbone[3 + i]          # .depthwise / .pointwise Sequentials
        if stride == 2:
            x = tf.keras.layers.ZeroPadding2D(1, name=f"ds{i}_pad")(x)
            dw_padding = "valid"
        else:
            # stride 1, k=3: SAME == symmetric +-1 padding
            dw_padding = "same"
        dw = tf.keras.layers.DepthwiseConv2D(
            3, strides=stride, padding=dw_padding, use_bias=False,
            name=f"ds{i}_dw"); x = dw(x); _set_dw(dw, pt_block.depthwise[0])
        dbn = _bn(f"ds{i}_dw_bn"); x = dbn(x); _set_bn(dbn, pt_block.depthwise[1])
        x = tf.keras.layers.Activation(_hard_swish, name=f"ds{i}_dw_hsw")(x)
        pw = tf.keras.layers.Conv2D(
            cout, 1, strides=1, padding="valid", use_bias=False,
            name=f"ds{i}_pw"); x = pw(x); _set_conv(pw, pt_block.pointwise[0])
        pbn = _bn(f"ds{i}_pw_bn"); x = pbn(x); _set_bn(pbn, pt_block.pointwise[1])
        x = tf.keras.layers.Activation(_hard_swish, name=f"ds{i}_pw_hsw")(x)

    # Head: GAP -> Dense(64)+ReLU -> Dropout(skip) -> Dense(1)+Sigmoid
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    d1 = tf.keras.layers.Dense(64, activation="relu", name="fc1")
    x = d1(x); _set_dense(d1, pt.head[2])
    d2 = tf.keras.layers.Dense(1, activation="sigmoid", name="fc2")
    out = d2(x); _set_dense(d2, pt.head[5])

    return tf.keras.Model(inp, out, name="srm_forensic_cnn")


# --------------------------------------------------------------------------
# LCNNMFMAudio
# --------------------------------------------------------------------------

def build_keras_lcnn(pt_model, input_frames: int, n_lfcc: int = 40) -> tf.keras.Model:
    """Rebuild LCNN-MFM channels-last: input (frames, n_lfcc, 1)."""
    pt = pt_model

    inp = tf.keras.Input((input_frames, n_lfcc, 1), name="input")

    def conv_block(x, pt_seq, idx, filters, kernel, name):
        c = tf.keras.layers.Conv2D(filters, kernel, padding="same", name=name)
        x = c(x); _set_conv(c, pt_seq[idx])
        m = _MFM(); x = m(x)
        return x

    # conv1 + MFM + pool            (1, T, F) -> (64, T, F) -> 32 -> pooled
    x = conv_block(inp, pt.conv1, 0, 64, 5, "lcnn_conv1")
    x = tf.keras.layers.MaxPooling2D(2, 2, name="pool1")(x)

    for bi, (ch, ksize) in enumerate([(96, 1), (96, 3)], start=1):
        x = conv_block(x, pt.block1, 2 * (bi - 1), ch, ksize, f"b1_c{bi}")
    x = tf.keras.layers.MaxPooling2D(2, 2, name="pool2")(x)

    for bi, (ch, ksize) in enumerate([(128, 1), (128, 3)], start=1):
        x = conv_block(x, pt.block2, 2 * (bi - 1), ch, ksize, f"b2_c{bi}")
    x = tf.keras.layers.MaxPooling2D(2, 2, name="pool3")(x)

    for bi, (ch, ksize) in enumerate([(160, 1), (160, 3)], start=1):
        x = conv_block(x, pt.block3, 2 * (bi - 1), ch, ksize, f"b3_c{bi}")
    x = tf.keras.layers.MaxPooling2D(2, 2, name="pool4")(x)

    for bi, (ch, ksize) in enumerate([(192, 1), (192, 3)], start=1):
        x = conv_block(x, pt.block4, 2 * (bi - 1), ch, ksize, f"b4_c{bi}")

    # AdaptiveAvgPool2d((1, 2)) == mean over time (keep W), then Flatten the
    # surviving 2 frequency groups => W*C channels. Here C=96, W=2 => 192.
    # Average over time (axis=1 in channels-last NHWC (B,T,W,C)); NOT
    # GlobalAveragePooling2D which also collapses W.
    # (Lambda/Permute layers: raw tf.* ops on symbolic KerasTensors are
    # rejected by Keras 3, so every op lives inside a layer.)
    x = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1), name="time_mean")(x)  # (B, W=2, C=96)
    # Cross-platform flatten-order contract: PyTorch flattens (B,C,1,2) as
    # c*2+w, so Keras must present (B, C, W) for the Dense weights to line up.
    x = tf.keras.layers.Permute((2, 1), name="to_channel_first")(x)  # (B, C=96, W=2)
    x = tf.keras.layers.Flatten(name="gap_flat")(x)  # (B, C*W) = (B, 192)

    fc1 = tf.keras.layers.Dense(160, name="cls_fc1")
    x = fc1(x); _set_dense(fc1, pt.classifier[0])
    x = _MFM()(x)
    # NOTE: pt.classifier[2] is Dropout (train-time only, no weights).
    fc2 = tf.keras.layers.Dense(1, activation="sigmoid", name="cls_fc2")
    out = fc2(x); _set_dense(fc2, pt.classifier[3])

    return tf.keras.Model(inp, out, name="lcnn_mfm_audio")


# --------------------------------------------------------------------------
# conversion entry points
# --------------------------------------------------------------------------

def keras_parity_check(keras_model, pt_model, inputs_nhwc, modality: str, tolerance: float = 1e-4) -> float:
    """Max |keras - pytorch| over NHWC inputs. Catches weight-transfer bugs
    independently of quantization error."""
    import torch

    if len(inputs_nhwc) == 0:
        raise ValueError("FP32 parity needs actual inputs")
    worst = 0.0
    pt_model.cpu().eval()
    for x in inputs_nhwc[:32]:
        xb = x[np.newaxis]
        if not np.isfinite(xb).all():
            raise ValueError('Non-finite FP32 parity input')
        k_values = np.asarray(keras_model(xb, training=False).numpy())
        with torch.no_grad():
            p_values = pt_model(torch.from_numpy(from_keras_input(xb, modality))).numpy()
        if k_values.shape != (1, 1) or p_values.shape != (1, 1) or not np.isfinite(k_values).all() or not np.isfinite(p_values).all():
            raise ValueError('FP32 parity requires one finite scalar score from each model')
        k_out, p_out = float(k_values[0, 0]), float(p_values[0, 0])
        worst = max(worst, abs(k_out - p_out))
    print(f"FP32 Keras-vs-PyTorch parity ({modality}): max err {worst:.2e}")
    if not np.isfinite(worst) or worst > tolerance:
        raise ValueError(f"FP32 weight-transfer parity failed: {worst} > {tolerance}")
    return worst


def keras_to_tflite(keras_model, output_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tfl = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tfl)
    size_kb = len(tfl) / 1024
    print(f"TFLite written: {output_path} ({size_kb:.0f} KB)")
    return output_path


def to_keras_input(x_pt_layout: np.ndarray, modality: str) -> np.ndarray:
    """(N,1?,C,H,W)/(N,1,T,F) PT tensors -> NHWC float32 for TF."""
    x = np.asarray(x_pt_layout, dtype=np.float32)
    if modality == "visual":
        if x.ndim == 4:      # (N,C,H,W)
            return np.transpose(x, (0, 2, 3, 1))
        return np.transpose(x, (1, 2, 0))          # (H,W,C)
    if modality == "audio":
        if x.ndim == 4:      # (N,1,T,F)
            return np.transpose(x, (0, 2, 3, 1))   # (N,T,F,1)
        if x.ndim == 3:      # (1,T,F)
            return np.transpose(x, (1, 2, 0))[None]
        return x[..., None][None]                  # (T,F,1)
    raise ValueError(modality)


def from_keras_input(x_nhwc: np.ndarray, modality: str) -> np.ndarray:
    """NHWC -> PT layout so the original model can score the same tensor."""
    x = np.asarray(x_nhwc, dtype=np.float32)
    if modality == "visual":
        if x.ndim == 4:
            return np.transpose(x, (0, 3, 1, 2))
        return np.transpose(x, (2, 0, 1))[None]
    if modality == "audio":
        if x.ndim == 4:
            return np.transpose(x, (0, 3, 1, 2))
        return np.transpose(x, (2, 0, 1))[None]
    raise ValueError(modality)
