# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午2:45
@file: test_random_choice.py
@author: zj
@description: 
"""

from scipy import stats
import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_random_choice():
    random_state = random.getstate()
    random.seed(42)
    random_choice_transform = transforms.RandomChoice(
        [
            transforms.Resize(15),
            transforms.Resize(20),
            transforms.CenterCrop(10)
        ]
    )
    # img = transforms.ToPILImage()(torch.rand(3, 25, 25))
    img = transforms.ToNumpyImage()(torch.rand(3, 25, 25))
    num_samples = 250
    num_resize_15 = 0
    num_resize_20 = 0
    num_crop_10 = 0
    for _ in range(num_samples):
        out = random_choice_transform(img)
        # if out.size == (15, 15):
        if out.shape[:2] == (15, 15):
            num_resize_15 += 1
        # elif out.size == (20, 20):
        elif out.shape[:2] == (20, 20):
            num_resize_20 += 1
        # elif out.size == (10, 10):
        elif out.shape[:2] == (10, 10):
            num_crop_10 += 1

    p_value = stats.binom_test(num_resize_15, num_samples, p=0.33333)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001
    p_value = stats.binom_test(num_resize_20, num_samples, p=0.33333)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001
    p_value = stats.binom_test(num_crop_10, num_samples, p=0.33333)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    random.setstate(random_state)
    # Checking if RandomChoice can be printed as string
    random_choice_transform.__repr__()
