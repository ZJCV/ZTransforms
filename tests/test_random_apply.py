# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午2:27
@file: test_random_apply.py
@author: zj
@description: 
"""

from scipy import stats
import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_random_apply():
    random_state = random.getstate()
    random.seed(42)
    random_apply_transform = transforms.RandomApply(
        [
            transforms.RandomRotation((-45, 45)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ], p=0.75
    )
    # img = transforms.ToPILImage()(torch.rand(3, 10, 10))
    img = transforms.ToNumpyImage()(torch.rand(3, 10, 10))
    num_samples = 250
    num_applies = 0
    for _ in range(num_samples):
        out = random_apply_transform(img)
        # if out != img:
        if not (out == img).all():
            num_applies += 1

    p_value = stats.binom_test(num_applies, num_samples, p=0.75)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    # Checking if RandomApply can be printed as string
    random_apply_transform.__repr__()
