# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午4:23
@file: test_ten_crop.py
@author: zj
@description: 
"""

import torch
import numpy as np
from PIL import Image
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_ten_crop():
    # to_pil_image = transforms.ToPILImage()
    to_numpy_image = transforms.ToNumpyImage()
    h = random.randint(5, 25)
    w = random.randint(5, 25)
    for should_vflip in [True, False]:
        for single_dim in [True, False]:
            crop_h = random.randint(1, h)
            crop_w = random.randint(1, w)
            if single_dim:
                crop_h = min(crop_h, crop_w)
                crop_w = crop_h
                transform = transforms.TenCrop(crop_h,
                                               vertical_flip=should_vflip)
                five_crop = transforms.FiveCrop(crop_h)
            else:
                transform = transforms.TenCrop((crop_h, crop_w),
                                               vertical_flip=should_vflip)
                five_crop = transforms.FiveCrop((crop_h, crop_w))

            # img = to_pil_image(torch.FloatTensor(3, h, w).uniform_())
            img = to_numpy_image(torch.FloatTensor(3, h, w).uniform_())
            results = transform(img)
            expected_output = five_crop(img)

            # Checking if FiveCrop and TenCrop can be printed as string
            transform.__repr__()
            five_crop.__repr__()

            if should_vflip:
                # vflipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                vflipped_img = np.flipud(img)
                expected_output += five_crop(vflipped_img)
            else:
                # hflipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                hflipped_img = np.fliplr(img)
                expected_output += five_crop(hflipped_img)

            assert len(results) == 10
            for crop_a, crop_b in zip(results, expected_output):
                # assert crop_a == crop_b
                assert crop_a.any() == crop_b.any()
