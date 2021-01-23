# -*- coding: utf-8 -*-

"""
@date: 2021/1/23 下午2:15
@file: test_random_affine.py
@author: zj
@description: 
"""

import pytest
import random
import numpy as np
import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F


# import ztransforms.cls_transforms as transforms
# import ztransforms.cls_functional as F
import ztransforms.transforms as transforms
import ztransforms.functional as F


def test_random_affine():
    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        transforms.RandomAffine(-0.7)
        transforms.RandomAffine([-0.7])
        transforms.RandomAffine([-0.7, 0, 0.7])

        transforms.RandomAffine([-90, 90], translate=2.0)
        transforms.RandomAffine([-90, 90], translate=[-1.0, 1.0])
        transforms.RandomAffine([-90, 90], translate=[-1.0, 0.0, 1.0])

        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.0])
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[-1.0, 1.0])
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, -0.5])
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 3.0, -0.5])

        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=-7)
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10])
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10])
        transforms.RandomAffine([-90, 90], translate=[0.2, 0.2], scale=[0.5, 0.5], shear=[-10, 0, 10, 0, 10])

    x = np.zeros((100, 100, 3), dtype=np.uint8)
    img = F.to_pil_image(x)

    t = transforms.RandomAffine(10, translate=[0.5, 0.3], scale=[0.7, 1.3], shear=[-10, 10, 20, 40])
    for _ in range(100):
        angle, translations, scale, shear = t.get_params(t.degrees, t.translate, t.scale, t.shear,
                                                         img_size=img.size)
        # self.assertTrue(-10 < angle < 10)
        # self.assertTrue(-img.size[0] * 0.5 <= translations[0] <= img.size[0] * 0.5,
        #                 "{} vs {}".format(translations[0], img.size[0] * 0.5))
        # self.assertTrue(-img.size[1] * 0.5 <= translations[1] <= img.size[1] * 0.5,
        #                 "{} vs {}".format(translations[1], img.size[1] * 0.5))
        # self.assertTrue(0.7 < scale < 1.3)
        # self.assertTrue(-10 < shear[0] < 10)
        # self.assertTrue(-20 < shear[1] < 40)
        assert (-10 < angle < 10)
        assert (-img.size[0] * 0.5 <= translations[0] <= img.size[0] * 0.5,
                "{} vs {}".format(translations[0], img.size[0] * 0.5))
        assert (-img.size[1] * 0.5 <= translations[1] <= img.size[1] * 0.5,
                "{} vs {}".format(translations[1], img.size[1] * 0.5))
        assert (0.7 < scale < 1.3)
        assert (-10 < shear[0] < 10)
        assert (-20 < shear[1] < 40)

    # Checking if RandomAffine can be printed as string
    t.__repr__()

    t = transforms.RandomAffine(10, interpolation=transforms.InterpolationMode.BILINEAR)
    # self.assertIn("bilinear", t.__repr__())
    assert "bilinear" in t.__repr__()

    # assert deprecation warning and non-BC
    # with self.assertWarnsRegex(UserWarning, r"Argument resample is deprecated and will be removed"):
    with pytest.warns(UserWarning, match=r"Argument resample is deprecated and will be removed"):
        t = transforms.RandomAffine(10, resample=2)
        # self.assertEqual(t.interpolation, transforms.InterpolationMode.BILINEAR)
        assert t.interpolation == transforms.InterpolationMode.BILINEAR

    # with self.assertWarnsRegex(UserWarning, r"Argument fillcolor is deprecated and will be removed"):
    with pytest.warns(UserWarning, match=r"Argument fillcolor is deprecated and will be removed"):
        t = transforms.RandomAffine(10, fillcolor=10)
        # self.assertEqual(t.fill, 10)
        assert t.fill == 10

    # assert changed type warning
    # with self.assertWarnsRegex(UserWarning, r"Argument interpolation should be of type InterpolationMode"):
    with pytest.warns(UserWarning, match=r"Argument interpolation should be of type InterpolationMode"):
        t = transforms.RandomAffine(10, interpolation=2)
        # self.assertEqual(t.interpolation, transforms.InterpolationMode.BILINEAR)
        assert t.interpolation == transforms.InterpolationMode.BILINEAR
