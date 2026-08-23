import torch
import torch.nn as nn
import os

class SRMLayer(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) Filter layer to extract high-frequency noise residuals.
    These filters are effective at highlighting artifacts introduced by generative AI.
    """
    def __init__(self):
        super(SRMLayer, self).__init__()
        # 3 output channels, 3 input channels, 5x5 kernel
        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False)
        
        # Define 5x5 filters
        # 1. Laplacian-like center-surround
        f1 = torch.tensor([
            [-1, -1, -1, -1, -1],
            [-1,  1,  2,  1, -1],
            [-1,  2,  4,  2, -1],
            [-1,  1,  2,  1, -1],
            [-1, -1, -1, -1, -1]
        ], dtype=torch.float32)
        f1 = f1 / 4.0 # Normalize roughly

        # 2. Horizontal edge/noise detector
        f2 = torch.tensor([
            [-1, -2,  0,  2,  1],
            [-2, -4,  0,  4,  2],
            [-4, -8,  0,  8,  4],
            [-2, -4,  0,  4,  2],
            [-1, -2,  0,  2,  1]
        ], dtype=torch.float32)
        f2 = f2 / 8.0

        # 3. Diagonal high-frequency detector
        f3 = torch.tensor([
            [-1, -1,  0,  1,  1],
            [-1, -2,  0,  2,  1],
            [ 0,  0,  0,  0,  0],
            [ 1,  2,  0, -2, -1],
            [ 1,  1,  0, -1, -1]
        ], dtype=torch.float32)
        f3 = f3 / 4.0

        # Assign to weights (out_channels, in_channels, kH, kW)
        # We divide by 3 to average the contribution of RGB channels
        weight = torch.zeros((3, 3, 5, 5), dtype=torch.float32)
        for in_c in range(3):
            weight[0, in_c, :, :] = f1 / 3.0
            weight[1, in_c, :, :] = f2 / 3.0
            weight[2, in_c, :, :] = f3 / 3.0
            
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
    Lightweight CNN for detecting AI-generated images via noise residual analysis.
    Designed for on-device inference with highly restricted parameter count.
    """
    def __init__(self):
        super(SRMForensicCNN, self).__init__()
        
        # 1. SRM Filter Layer (Frozen)
        self.srm = SRMLayer()
        
        # 2. CNN Backbone (MobileNetV3-inspired)
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            
            # Block 2
            DepthwiseSeparableBlock(16, 32, stride=1),
            
            # Block 3
            DepthwiseSeparableBlock(32, 64, stride=2),
            
            # Block 4
            DepthwiseSeparableBlock(64, 128, stride=2),
            
            # Block 5
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 224, 224)
        Returns:
            torch.Tensor: Probabilities of shape (B, 1) [0=real, 1=AI-generated]
        """
        x = self.srm(x)
        x = self.backbone(x)
        x = self.head(x)
        return x

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
        
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Check if it's a Lightning checkpoint (often nested under 'state_dict')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
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
