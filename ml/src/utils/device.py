import torch

def get_device() -> torch.device:
    """Returns the most optimal device available (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_gpu_info() -> dict:
    """Returns information about the available GPU, if any."""
    info = {
        "name": "None",
        "vram_gb": 0.0,
        "compute_capability": "N/A"
    }
    if torch.cuda.is_available():
        info["name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
    return info
