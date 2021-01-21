# -*- coding: utf-8 -*-

"""
@date: 2021/1/21 下午4:27
@file: test_random_horizontal_flip.py
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


def test_random_horizontal_flip():
    random_state = random.getstate()
    random.seed(42)
    # img = transforms.ToPILImage()(torch.rand(3, 10, 10))
    img = transforms.ToNumpyImage()(torch.rand(3, 10, 10))
    # himg = img.transpose(Image.FLIP_LEFT_RIGHT)
    himg = A.HorizontalFlip(always_apply=True).apply(img)

    num_samples = 250
    num_horizontal = 0
    for _ in range(num_samples):
        out = transforms.RandomHorizontalFlip()(img)
        # if out == himg:
        if (out == himg).all():
            num_horizontal += 1

    p_value = stats.binom_test(num_horizontal, num_samples, p=0.5)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    num_samples = 250
    num_horizontal = 0
    for _ in range(num_samples):
        out = transforms.RandomHorizontalFlip(p=0.7)(img)
        # if out == himg:
        if (out == himg).all():
            num_horizontal += 1

    p_value = stats.binom_test(num_horizontal, num_samples, p=0.7)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    # Checking if RandomHorizontalFlip can be printed as string
    transforms.RandomHorizontalFlip().__repr__()
