# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午2:48
@file: test_random_order.py
@author: zj
@description: 
"""

from scipy import stats
import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_random_order():
    random_state = random.getstate()
    random.seed(42)
    random_order_transform = transforms.RandomOrder(
        [
            transforms.Resize(20),
            transforms.CenterCrop(10)
        ]
    )
    # img = transforms.ToPILImage()(torch.rand(3, 25, 25))
    img = transforms.ToNumpyImage()(torch.rand(3, 25, 25))
    num_samples = 250
    num_normal_order = 0
    resize_crop_out = transforms.CenterCrop(10)(transforms.Resize(20)(img))
    for _ in range(num_samples):
        out = random_order_transform(img)
        # if out == resize_crop_out:
        if out.shape == resize_crop_out.shape and (out == resize_crop_out).all():
            num_normal_order += 1

    p_value = stats.binom_test(num_normal_order, num_samples, p=0.5)
    random.setstate(random_state)
    # self.assertGreater(p_value, 0.0001)
    assert p_value > 0.0001

    # Checking if RandomOrder can be printed as string
    random_order_transform.__repr__()
