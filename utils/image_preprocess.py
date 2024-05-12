from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch

normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
                                 std=[0.260, 0.253, 0.271])

def transforms_train_val():
    """
        You can modify the train_transforms to try different image preprocessing methods when training model
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize([84, 84]),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms

def transforms_test():
    """
        You can modify the function to try different image fusion methods when evaluating the trained model
    """
    trans = transforms.Compose([
        transforms.Resize([84, 84]),
        transforms.ToTensor(),
        normalize,
    ])
    return trans