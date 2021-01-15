# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午3:53
@file: test_five_crop.py
@author: zj
@description: 
"""

import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_five_crop():
    # to_pil_image = transforms.ToPILImage()
    to_numpy_image = transforms.ToNumpyImage()
    h = random.randint(5, 25)
    w = random.randint(5, 25)
    for single_dim in [True, False]:
        crop_h = random.randint(1, h)
        crop_w = random.randint(1, w)
        if single_dim:
            crop_h = min(crop_h, crop_w)
            crop_w = crop_h
            transform = transforms.FiveCrop(crop_h)
        else:
            transform = transforms.FiveCrop((crop_h, crop_w))

        img = torch.FloatTensor(3, h, w).uniform_()
        # results = transform(to_pil_image(img))
        results = transform(to_numpy_image(img))

        assert len(results) == 5
        for crop in results:
            assert crop.shape[:2] == (crop_h, crop_w)

        # to_pil_image = transforms.ToPILImage()
        # tl = to_pil_image(img[:, 0:crop_h, 0:crop_w])
        # tr = to_pil_image(img[:, 0:crop_h, w - crop_w:])
        # bl = to_pil_image(img[:, h - crop_h:, 0:crop_w])
        # br = to_pil_image(img[:, h - crop_h:, w - crop_w:])
        tl = to_numpy_image(img[:, 0:crop_h, 0:crop_w])
        tr = to_numpy_image(img[:, 0:crop_h, w - crop_w:])
        bl = to_numpy_image(img[:, h - crop_h:, 0:crop_w])
        br = to_numpy_image(img[:, h - crop_h:, w - crop_w:])
        # center = transforms.CenterCrop((crop_h, crop_w))(to_pil_image(img))
        center = transforms.CenterCrop((crop_h, crop_w))(to_numpy_image(img))
        expected_output = (tl, tr, bl, br, center)
        assert len(results) == len(expected_output)
        for crop_a, crop_b in zip(results, expected_output):
            assert crop_a.any() == crop_b.any()


if __name__ == '__main__':
    test_five_crop()
