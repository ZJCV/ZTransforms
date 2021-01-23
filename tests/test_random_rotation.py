# -*- coding: utf-8 -*-

"""
@date: 2021/1/22 下午8:29
@file: test_random_rotation.py
@author: zj
@description: 
"""

import pytest
# import torchvision.transforms as transforms
import ztransforms.transforms as transforms


def test_random_rotation():
    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        transforms.RandomRotation(-0.7)
        transforms.RandomRotation([-0.7])
        transforms.RandomRotation([-0.7, 0, 0.7])

    t = transforms.RandomRotation(10)
    angle = t.get_params(t.degrees)
    # self.assertTrue(angle > -10 and angle < 10)
    assert (angle > -10 and angle < 10)

    t = transforms.RandomRotation((-10, 10))
    angle = t.get_params(t.degrees)
    # self.assertTrue(-10 < angle < 10)
    assert (-10 < angle < 10)

    # Checking if RandomRotation can be printed as string
    t.__repr__()

    # assert deprecation warning and non-BC
    # with self.assertWarnsRegex(UserWarning, r"Argument resample is deprecated and will be removed"):
    with pytest.warns(UserWarning, match=r"Argument resample is deprecated and will be removed"):
        t = transforms.RandomRotation((-10, 10), resample=2)
        # self.assertEqual(t.interpolation, transforms.InterpolationMode.BILINEAR)
        assert t.interpolation == transforms.InterpolationMode.BILINEAR

    # assert changed type warning
    # with self.assertWarnsRegex(UserWarning, r"Argument interpolation should be of type InterpolationMode"):
    with pytest.warns(UserWarning, match=r"Argument interpolation should be of type InterpolationMode"):
        t = transforms.RandomRotation((-10, 10), interpolation=2)
        # self.assertEqual(t.interpolation, transforms.InterpolationMode.BILINEAR)
        assert t.interpolation == transforms.InterpolationMode.BILINEAR
