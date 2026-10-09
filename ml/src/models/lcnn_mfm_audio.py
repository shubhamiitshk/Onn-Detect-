import torch
import torch.nn as nn
import os

class MaxFeatureMap(nn.Module):
    """
    Max Feature Map (MFM) activation function.
    Splits the feature maps in half along a specified dimension and takes the element-wise maximum.
    Acts as a natural feature selector and noise filter.
    """
    def __init__(self, mfm_type: str = '2d'):
        super(MaxFeatureMap, self).__init__()
        assert mfm_type in ['1d', '2d'], "mfm_type must be either '1d' or '2d'"
        self.mfm_type = mfm_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D case: input shape (B, C, H, W). Split along channels (dim=1).
        # 1D case: input shape (B, F). Split along features (dim=1).
        out = torch.chunk(x, 2, dim=1)
        return torch.max(out[0], out[1])


class LCNNMFMAudio(nn.Module):
    """
    Light CNN with Max Feature Map (MFM) activation for audio deepfake detection.
    Optimized for ASVspoof-style tasks, expecting LFCC spectrograms as input.
    """
    def __init__(self):
        super(LCNNMFMAudio, self).__init__()
        
        # Initial Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MaxFeatureMap('2d') # 64 -> 32 channels
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=1),
            MaxFeatureMap('2d'), # 96 -> 48 channels
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            MaxFeatureMap('2d')  # 96 -> 48 channels
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=1),
            MaxFeatureMap('2d'), # 128 -> 64 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            MaxFeatureMap('2d')  # 128 -> 64 channels
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 160, kernel_size=1),
            MaxFeatureMap('2d'), # 160 -> 80 channels
            nn.Conv2d(80, 160, kernel_size=3, padding=1),
            MaxFeatureMap('2d')  # 160 -> 80 channels
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(80, 192, kernel_size=1),
            MaxFeatureMap('2d'), # 192 -> 96 channels
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            MaxFeatureMap('2d')  # 192 -> 96 channels
        )
        
        # Global Pooling and Classifier
        # Pool over TIME only, keeping the reduced frequency extent. The old
        # AdaptiveAvgPool2d(1) averaged over BOTH axes, which collapses the
        # cepstral ordering (c0 = log energy, c1..c39 = spectral shape) into
        # one number per channel - the model literally cannot see where in
        # frequency a vocoder artifact lives. Pooling time to 1 and keeping
        # frequency preserves the discrimination the LFCC front end encodes.
        # Input contract is (T, 40): after 4 x stride-2 pools the frequency
        # extent is 40 -> 20 -> 10 -> 5 -> 2, so we keep exactly 2 groups and
        # halve nothing away.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 2))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            # 96 channels x 2 frequency groups
            nn.Linear(96 * 2, 160),
            MaxFeatureMap('1d'), # 160 -> 80 features
            # Dropout on the MFM output (STC recipe: 0.4). Inactive at export:
            # the Keras builder transfers only the two Linear layers, so the
            # indices below must skip this position (see torch_to_tflite_keras
            # build_keras_lcnn: fc1 <- classifier[0], fc2 <- classifier[3]).
            nn.Dropout(p=0.4),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): LFCC input of shape (B, 1, time_frames, n_lfcc_coeffs)
        Returns:
            torch.Tensor: Probability of being synthetic (B, 1)
        """
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.block1(x)
        x = self.pool2(x)
        
        x = self.block2(x)
        x = self.pool3(x)
        
        x = self.block3(x)
        x = self.pool4(x)
        
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        
        x = self.classifier[:-1](x) if return_logits else self.classifier(x)
        
        return x

    def forward_logits(self, x):
        return self.forward(x, return_logits=True)

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "LCNNMFMAudio":
        """Loads a pretrained model from a specified path."""
        model = cls()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        from training.checkpoint import load_verified_checkpoint
        checkpoint = load_verified_checkpoint(checkpoint_path)
        if checkpoint['contract'].get('architecture') != cls.__name__:
            raise ValueError('Checkpoint architecture mismatch')
        from data.lfcc_frontend import FRONTEND_ID
        if checkpoint['contract']['frontend_id'] != FRONTEND_ID:
            raise ValueError('Checkpoint LFCC frontend mismatch')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        return model

if __name__ == "__main__":
    model = LCNNMFMAudio()
    
    # Batch=1, Channel=1, Time=400, LFCC_coeffs=40
    dummy_input = torch.randn(1, 1, 400, 40)
    out = model(dummy_input)
    
    total_params = model.count_parameters()
    fp32_size = (total_params * 4) / (1024 ** 2)
    int8_size = total_params / (1024 ** 2)
    
    print("="*50)
    print("LCNN-MFM Audio Model Summary")
    print("="*50)
    print(f"Output Shape:      {out.shape}")
    print(f"Trainable Params:  {total_params:,}")
    print(f"Est. Size (FP32):  {fp32_size:.2f} MB")
    print(f"Est. Size (INT8):  {int8_size:.2f} MB")
    print("="*50)
