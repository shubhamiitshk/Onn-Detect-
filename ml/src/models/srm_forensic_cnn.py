import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class SRMLayer(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) + Optional Neighboring Pixel Relationships (NPR)
    + Optional RGB/Luminance Low-Pass stem (`out_channels` in {12, 16, 20}).

    - 12 channels (`include_npr=False, include_rgb=False`): Legacy SRM high-pass filters.
    - 16 channels (`include_npr=True, include_rgb=False`): SRM + 4 NPR 2x2 upsampling
      residual filters (Tan et al., CVPR 2024).
    - 20 channels (`include_npr=True, include_rgb=True`): SRM + NPR + 3 centered RGB
      pass-through filters + 1 5x5 Gaussian/box luminance low-pass filter so the
      MobileNetV3 backbone sees both high-frequency noise residuals and low/mid-frequency
      RGB color/lighting structure during vision-tower distillation.

    Fully compatible with `build_keras_srm` in `ml/src/export/torch_to_tflite_keras.py`,
    which reads `srm_out = int(pt.srm.conv.out_channels)` dynamically.
    """
    def __init__(self, include_npr: bool = False, include_rgb: bool = False):
        super(SRMLayer, self).__init__()
        self.include_npr = include_npr
        self.include_rgb = include_rgb
        out_channels = 12 + (4 if include_npr else 0) + (4 if include_rgb else 0)
        self.conv = nn.Conv2d(3, out_channels, kernel_size=5, padding=2, bias=False)

        # 1. 5x5 Laplacian-like center-surround
        f1 = torch.tensor([
            [-1, -1, -1, -1, -1],
            [-1,  1,  2,  1, -1],
            [-1,  2,  4,  2, -1],
            [-1,  1,  2,  1, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=torch.float32) / 4.0

        # 2. 5x5 Horizontal edge/noise detector
        f2 = torch.tensor([
            [-1, -2,  0,  2,  1],
            [-2, -4,  0,  4,  2],
            [-4, -8,  0,  8,  4],
            [-2, -4,  0,  4,  2],
            [-1, -2,  0,  2,  1]
        ], dtype=torch.float32) / 8.0

        # 3. 5x5 Vertical edge/noise detector
        f2_v = f2.t().contiguous()

        # 4. 5x5 Main diagonal high-frequency detector
        f3 = torch.tensor([
            [-1, -1,  0,  1,  1],
            [-1, -2,  0,  2,  1],
            [ 0,  0,  0,  0,  0],
            [ 1,  2,  0, -2, -1],
            [ 1,  1,  0, -1, -1]
        ], dtype=torch.float32) / 4.0

        # 5. 5x5 Anti-diagonal high-frequency detector
        f3_anti = torch.flip(f3, dims=[1]).contiguous()

        # 6. 5x5 Fridrich KV 4th-order steganalysis filter
        f_kv = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=torch.float32) / 12.0

        # 7. Nyquist checkerboard filter (detects VAE/upsampling grid periodicity)
        f_chk = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0, -2,  4, -2,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0

        # 3x3 Laplacian embedded in 5x5 for per-channel and color-opponent residuals
        f_lap3 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0, -1,  4, -1,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=torch.float32) / 4.0

        weight = torch.zeros((out_channels, 3, 5, 5), dtype=torch.float32)
        luma_filters = [f1, f2, f2_v, f3, f3_anti, f_kv, f_chk]
        for idx, filt in enumerate(luma_filters):
            for in_c in range(3):
                weight[idx, in_c, :, :] = filt / 3.0

        # Per-channel R, G, B high-pass residuals
        weight[7, 0, :, :] = f_lap3
        weight[8, 1, :, :] = f_lap3
        weight[9, 2, :, :] = f_lap3

        # Color-opponent chrominance high-pass residuals (R - G and B - G)
        weight[10, 0, :, :] = 0.5 * f_lap3
        weight[10, 1, :, :] = -0.5 * f_lap3
        weight[11, 2, :, :] = 0.5 * f_lap3
        weight[11, 1, :, :] = -0.5 * f_lap3

        cursor = 12
        if include_npr:
            # 12..15: Neighboring Pixel Relationships (NPR) 2x2 sub-grid operators
            npr_h = torch.zeros((5, 5), dtype=torch.float32)
            npr_h[2, 2], npr_h[2, 3] = 1.0, -1.0
            npr_v = torch.zeros((5, 5), dtype=torch.float32)
            npr_v[2, 2], npr_v[3, 2] = 1.0, -1.0
            npr_d1 = torch.zeros((5, 5), dtype=torch.float32)
            npr_d1[2, 2], npr_d1[3, 3] = 1.0, -1.0
            npr_d2 = torch.zeros((5, 5), dtype=torch.float32)
            npr_d2[2, 3], npr_d2[3, 2] = 1.0, -1.0
            for offset, npr_k in enumerate([npr_h, npr_v, npr_d1, npr_d2]):
                for in_c in range(3):
                    weight[cursor + offset, in_c, :, :] = npr_k / 3.0
            cursor += 4

        if include_rgb:
            # 3 centered RGB pass-through identity kernels + 1 5x5 low-pass luminance kernel
            # Preserves low/mid-frequency color and shading through tanh(conv(x))
            for rgb_c in range(3):
                weight[cursor + rgb_c, rgb_c, 2, 2] = 1.0
            gauss_5x5 = torch.tensor([
                [1,  4,  6,  4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1,  4,  6,  4, 1],
            ], dtype=torch.float32) / 256.0
            for in_c in range(3):
                weight[cursor + 3, in_c, :, :] = gauss_5x5 / 3.0

        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(x))


class DepthwiseSeparableBlock(nn.Module):
    """MobileNetV3-inspired Depthwise Separable Convolution Block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableBlock, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SRMForensicCNN(nn.Module):
    """
    Lightweight CNN for detecting AI-generated images via noise, chrominance,
    Neighboring Pixel Relationship (NPR) residuals, and optional RGB low-pass cues.

    Auto-adapts its stem and SRM filter bank between 12 channels (legacy SRM),
    16 channels (SRM + NPR), and 20 channels (SRM + NPR + RGB/Luma) inside
    `load_state_dict()` so callers like `export_all_models.py` work seamlessly.
    """
    def __init__(self, include_npr: bool = False, include_rgb: bool = False):
        super(SRMForensicCNN, self).__init__()
        self._configure_channels(include_npr=include_npr, include_rgb=include_rgb)

        # 3. Classifier Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _configure_channels(self, include_npr: bool, include_rgb: bool = False) -> None:
        self.include_npr = include_npr
        self.include_rgb = include_rgb
        srm_channels = 12 + (4 if include_npr else 0) + (4 if include_rgb else 0)
        self.srm = SRMLayer(include_npr=include_npr, include_rgb=include_rgb)
        self.backbone = nn.Sequential(
            nn.Conv2d(srm_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            DepthwiseSeparableBlock(32, 32, stride=1),
            DepthwiseSeparableBlock(32, 64, stride=2),
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128, stride=1),
            DepthwiseSeparableBlock(128, 256, stride=2)
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        srm_weight = state_dict.get("srm.conv.weight")
        if srm_weight is not None:
            ch = int(srm_weight.shape[0])
            want_npr = ch in (16, 20)
            want_rgb = ch == 20
            if want_npr != self.include_npr or want_rgb != self.include_rgb:
                device = next(self.parameters()).device
                self._configure_channels(include_npr=want_npr, include_rgb=want_rgb)
                self.to(device)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x = self.srm(x)
        x = self.backbone(x)
        x = self.head[:-1](x) if return_logits else self.head(x)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_logits=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "SRMForensicCNN":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        from training.checkpoint import load_verified_checkpoint
        checkpoint = load_verified_checkpoint(checkpoint_path)
        if checkpoint["contract"].get("architecture") != cls.__name__:
            raise ValueError("Checkpoint architecture mismatch")
        state_dict = checkpoint["state_dict"]
        srm_weight = state_dict.get("srm.conv.weight")
        ch = int(srm_weight.shape[0]) if srm_weight is not None else 12
        model = cls(include_npr=ch in (16, 20), include_rgb=(ch == 20))
        model.load_state_dict(state_dict)
        return model


def extract_high_low_texture_patches(
    img: Image.Image,
    patch_size: int = 128,
    grid_stride: int = 64,
) -> tuple[Image.Image, Image.Image]:
    """PatchCraft-inspired native-resolution patch selector.

    Extracts the highest-gradient (rich texture) and lowest-gradient (smooth background)
    native-resolution crops BEFORE any global bilinear downsampling so high-frequency
    SRM/NPR synthesis residuals are preserved.
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < patch_size or h < patch_size:
        resized = rgb.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        return resized, resized

    arr = np.asarray(rgb, dtype=np.float32)
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = np.abs(np.diff(luma, axis=1, append=luma[:, -1:]))
    gy = np.abs(np.diff(luma, axis=0, append=luma[-1:, :]))
    energy = gx + gy

    best_hi, best_lo = -1.0, float("inf")
    hi_box = lo_box = (0, 0, patch_size, patch_size)

    ys = list(range(0, h - patch_size + 1, grid_stride)) or [0]
    xs = list(range(0, w - patch_size + 1, grid_stride)) or [0]
    for y in ys:
        for x in xs:
            score = float(energy[y : y + patch_size, x : x + patch_size].mean())
            box = (x, y, x + patch_size, y + patch_size)
            if score > best_hi:
                best_hi, hi_box = score, box
            if score < best_lo:
                best_lo, lo_box = score, box

    return rgb.crop(hi_box), rgb.crop(lo_box)