import torch
import torch.nn as nn
import os

class SRMLayer(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) Filter layer to extract high-frequency noise
    and color-opponent demosaicing/synthesis residuals.
    """
    def __init__(self):
        super(SRMLayer, self).__init__()
        # 12 output channels (7 luminance high-pass + 3 per-channel + 2 chroma-opponent), 3 input channels, 5x5 kernel
        self.conv = nn.Conv2d(3, 12, kernel_size=5, padding=2, bias=False)

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

        weight = torch.zeros((12, 3, 5, 5), dtype=torch.float32)
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

        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through the fixed filters and normalize to [-1, 1] using tanh
        return torch.tanh(self.conv(x))


class DepthwiseSeparableBlock(nn.Module):
    """
    MobileNetV3-inspired Depthwise Separable Convolution Block.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableBlock, self).__init__()
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True)
        )
        # Pointwise
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
    Lightweight CNN for detecting AI-generated images via noise & chrominance residual analysis.
    Designed for on-device inference with highly restricted parameter count.
    """
    def __init__(self):
        super(SRMForensicCNN, self).__init__()
        
        # 1. SRM Filter Layer (Frozen, 12 forensic channels)
        self.srm = SRMLayer()
        
        # 2. CNN Backbone (MobileNetV3-inspired)
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(12, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            
            # Stage 1
            DepthwiseSeparableBlock(32, 32, stride=1),
            DepthwiseSeparableBlock(32, 64, stride=2),
            
            # Stage 2
            DepthwiseSeparableBlock(64, 64, stride=1),
            DepthwiseSeparableBlock(64, 128, stride=2),
            
            # Stage 3
            DepthwiseSeparableBlock(128, 128, stride=1),
            DepthwiseSeparableBlock(128, 256, stride=2)
        )
        
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

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 224, 224)
        Returns:
            torch.Tensor: Probabilities of shape (B, 1) [0=real, 1=AI-generated]
        """
        x = self.srm(x)
        x = self.backbone(x)
        x = self.head[:-1](x) if return_logits else self.head(x)
        return x

    def forward_logits(self, x):
        return self.forward(x, return_logits=True)

    def count_parameters(self) -> int:
        """Counts the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "SRMForensicCNN":
        """
        Loads the model from a given checkpoint path.
        """
        model = cls()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        from training.checkpoint import load_verified_checkpoint
        checkpoint = load_verified_checkpoint(checkpoint_path)
        if checkpoint['contract'].get('architecture') != cls.__name__:
            raise ValueError('Checkpoint architecture mismatch')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        return model


if __name__ == "__main__":
    model = SRMForensicCNN()
    
    # Verify input/output shapes
    dummy_input = torch.randn(1, 3, 224, 224)
    out = model(dummy_input)
    
    # Calculate parameter count and sizes
    total_params = model.count_parameters()
    
    # FP32 takes 4 bytes per param, INT8 takes 1 byte per param
    fp32_size_mb = (total_params * 4) / (1024 ** 2)
    int8_size_mb = total_params / (1024 ** 2)
    
    print("="*50)
    print("SRM Forensic CNN Model Summary")
    print("="*50)
    print(model)
    print("="*50)
    print(f"Output Shape:      {out.shape}")
    print(f"Trainable Params:  {total_params:,}")
    print(f"Est. Size (FP32):  {fp32_size_mb:.2f} MB")
    print(f"Est. Size (INT8):  {int8_size_mb:.2f} MB")
    print("="*50)
