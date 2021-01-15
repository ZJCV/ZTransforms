# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午4:40
@file: test_randomresized_params.py
@author: zj
@description: 
"""

import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_randomresized_params():
    height = random.randint(24, 32) * 2
    width = random.randint(24, 32) * 2
    img = torch.ones(3, height, width)
    # to_pil_image = transforms.ToPILImage()
    to_numpy_image = transforms.ToNumpyImage()
    # img = to_pil_image(img)
    img = to_numpy_image(img)
    size = 100
    epsilon = 0.05
    min_scale = 0.25
    for _ in range(10):
        scale_min = max(round(random.random(), 2), min_scale)
        scale_range = (scale_min, scale_min + round(random.random(), 2))
        aspect_min = max(round(random.random(), 2), epsilon)
        aspect_ratio_range = (aspect_min, aspect_min + round(random.random(), 2))
        randresizecrop = transforms.RandomResizedCrop(size, scale_range, aspect_ratio_range)
        i, j, h, w = randresizecrop.get_params(img, scale_range, aspect_ratio_range)
        aspect_ratio_obtained = w / h
        # self.assertTrue((min(aspect_ratio_range) - epsilon <= aspect_ratio_obtained and
        #                  aspect_ratio_obtained <= max(aspect_ratio_range) + epsilon) or
        #                 aspect_ratio_obtained == 1.0)
        # self.assertIsInstance(i, int)
        # self.assertIsInstance(j, int)
        # self.assertIsInstance(h, int)
        # self.assertIsInstance(w, int)
        assert ((min(aspect_ratio_range) - epsilon <= aspect_ratio_obtained and
                 aspect_ratio_obtained <= max(aspect_ratio_range) + epsilon) or aspect_ratio_obtained == 1.0) == True
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(h, int)
        assert isinstance(w, int)
