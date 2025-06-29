import torch

def get_device():
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
