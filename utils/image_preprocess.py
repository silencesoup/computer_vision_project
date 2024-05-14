'''
Author: silencesoup silencesoup@outlook.com
Date: 2024-05-13 20:55:42
LastEditors: silencesoup silencesoup@outlook.com
LastEditTime: 2024-05-14 12:18:02
FilePath: /computer_vision_project/utils/image_preprocess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch

normalize = transforms.Normalize(mean=[0.484, 0.460, 0.411],
                                 std=[0.260, 0.253, 0.271])

from utils.cutout import Cutout

def transforms_train_val():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        Cutout(n_holes=1, length=16),  # Adding Cutout here
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