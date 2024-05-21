import torch

def label_smoothing(targets, num_classes, smoothing=0.1):
    """
    Apply label smoothing.
    
    Args:
        targets: Tensor of target labels.
        num_classes: Total number of classes.
        smoothing: Smoothing factor.
        
    Returns:
        Smoothed labels.
    """
    assert 0 <= smoothing < 1
    with torch.no_grad():
        targets = targets * (1 - smoothing) + smoothing / num_classes
    return targets