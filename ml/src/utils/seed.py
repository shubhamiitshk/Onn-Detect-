import random
import os
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """Seeds all random number generators to ensure reproducibility."""
    random.seed(seed)
    # PYTHONHASHSEED must be set before Python starts; runtime changes have no effect.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
