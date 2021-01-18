# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 上午9:22
@file: test_random_crop.py
@author: zj
@description: 
"""

import pytest
import numpy as np
import torch
import random
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_random_crop():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    oheight = random.randint(5, (height - 2) / 2) * 2
    owidth = random.randint(5, (width - 2) / 2) * 2
    img = torch.ones(3, height, width)
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.RandomCrop((oheight, owidth)),
        transforms.ToTensor(),
    ])(img)
    # self.assertEqual(result.size(1), oheight)
    # self.assertEqual(result.size(2), owidth)
    assert result.size(1) == oheight
    assert result.size(2) == owidth

    padding = random.randint(1, 20)
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.RandomCrop((oheight, owidth), padding=padding),
        transforms.ToTensor(),
    ])(img)
    # self.assertEqual(result.size(1), oheight)
    # self.assertEqual(result.size(2), owidth)
    assert result.size(1) == oheight
    assert result.size(2) == owidth

    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.RandomCrop((height, width)),
        transforms.ToTensor()
    ])(img)
    # self.assertEqual(result.size(1), height)
    # self.assertEqual(result.size(2), width)
    # self.assertTrue(np.allclose(img.numpy(), result.numpy()))
    assert result.size(1) == height
    assert result.size(2) == width
    # assert np.allclose(img.numpy(), result.numpy())
    assert np.allclose(img, result.numpy())

    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.RandomCrop((height + 1, width + 1), pad_if_needed=True),
        transforms.ToTensor(),
    ])(img)
    # self.assertEqual(result.size(1), height + 1)
    # self.assertEqual(result.size(2), width + 1)
    assert result.size(1) == (height + 1)
    assert result.size(2) == (width + 1)

    t = transforms.RandomCrop(48)
    img = torch.ones(3, 32, 32)
    # with self.assertRaisesRegex(ValueError, r"Required crop size .+ is larger then input image size .+"):
    #     t(img)
    with pytest.raises(ValueError, match=r"Required crop size .+ is larger then input image size .+"):
        t(img)
