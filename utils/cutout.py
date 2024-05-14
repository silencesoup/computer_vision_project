'''
Author: silencesoup silencesoup@outlook.com
Date: 2024-05-13 20:55:42
LastEditors: silencesoup silencesoup@outlook.com
LastEditTime: 2024-05-14 12:25:33
FilePath: /computer_vision_project/utils/cutout.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Apply cutout (zero patches) to the image
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def transforms_train_val():
    """
    Include CutMix and Cutout in training data preprocessing
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        Cutout(n_holes=1, length=16),  # You can adjust parameters
    ])
    val_transforms = transforms.Compose([
        transforms.Resize([84, 84]),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, val_transforms
