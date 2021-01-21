# -*- coding: utf-8 -*-

"""
@date: 2021/1/21 下午4:38
@file: test_random_vertical_flip.py
@author: zj
@description: 
"""

from scipy import stats
import random
from PIL import Image

import albumentations as A

import torch
# import torchvision.transforms as transforms
import ztransforms.transforms as transforms


def test_random_vertical_flip():
    random_state = random.getstate()
    random.seed(42)
    # img = transforms.ToPILImage()(torch.rand(3, 10, 10))
    img = transforms.ToNumpyImage()(torch.rand(3, 10, 10))
    # vimg = img.transpose(Image.FLIP_TOP_BOTTOM)
    vimg = A.VerticalFlip(always_apply=True).apply(img)

    num_samples = 250
    num_vertical = 0
    for _ in range(num_samples):
        out = transforms.RandomVerticalFlip()(img)
        # if out == vimg:
        if (out == vimg).all():
            num_vertical += 1

    p_value = stats.binom_test(num_vertical, num_samples, p=0.5)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    num_samples = 250
    num_vertical = 0
    for _ in range(num_samples):
        out = transforms.RandomVerticalFlip(p=0.7)(img)
        # if out == vimg:
        if (out == vimg).all():
            num_vertical += 1

    p_value = stats.binom_test(num_vertical, num_samples, p=0.7)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    # Checking if RandomVerticalFlip can be printed as string
    transforms.RandomVerticalFlip().__repr__()
