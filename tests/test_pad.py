# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 上午9:50
@file: test_pad.py
@author: zj
@description: 
"""

import pytest
import random
import numpy as np
import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F


import ztransforms.cls_transforms as transforms
import ztransforms.cls_functional as F


def test_pad():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    img = torch.ones(3, height, width)
    padding = random.randint(1, 20)
    fill = random.randint(1, 50)
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.Pad(padding, fill=fill),
        transforms.ToTensor(),
    ])(img)
    # self.assertEqual(result.size(1), height + 2 * padding)
    # self.assertEqual(result.size(2), width + 2 * padding)
    assert result.size(1) == (height + 2 * padding)
    assert result.size(2) == (width + 2 * padding)

    # check that all elements in the padded region correspond
    # to the pad value
    fill_v = fill / 255
    eps = 1e-5
    # self.assertTrue((result[:, :padding, :] - fill_v).abs().max() < eps)
    # self.assertTrue((result[:, :, :padding] - fill_v).abs().max() < eps)
    # self.assertRaises(ValueError, transforms.Pad(padding, fill=(1, 2)),
    #                   transforms.ToPILImage()(img))
    assert (result[:, :padding, :] - fill_v).abs().max() < eps
    assert (result[:, :, :padding] - fill_v).abs().max() < eps


def test_pad_with_tuple_of_pad_values():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    # img = transforms.ToPILImage()(torch.ones(3, height, width))
    img = transforms.ToNumpyImage()(torch.ones(3, height, width))

    padding = tuple([random.randint(1, 20) for _ in range(2)])
    output = transforms.Pad(padding)(img)
    # self.assertEqual(output.size, (width + padding[0] * 2, height + padding[1] * 2))
    # assert output.size == (width + padding[0] * 2, height + padding[1] * 2)
    assert output.shape[:2] == (height + padding[0] * 2, width + padding[1] * 2)

    padding = tuple([random.randint(1, 20) for _ in range(4)])
    output = transforms.Pad(padding)(img)
    # self.assertEqual(output.size[0], width + padding[0] + padding[2])
    # self.assertEqual(output.size[1], height + padding[1] + padding[3])
    # assert output.size[0] == width + padding[0] + padding[2]
    # assert output.size[1] == height + padding[1] + padding[3]
    assert output.shape[0] == height + padding[0] + padding[2]
    assert output.shape[1] == width + padding[1] + padding[3]

    # Checking if Padding can be printed as string
    transforms.Pad(padding).__repr__()


def test_pad_with_non_constant_padding_modes():
    """Unit tests for edge, reflect, symmetric padding"""
    img = torch.zeros(3, 27, 27).byte()
    img[:, :, 0] = 1  # Constant value added to leftmost edge
    # img = transforms.ToPILImage()(img)
    # img_a = np.array(img)
    img = transforms.ToNumpyImage()(img)
    # img = F.pad(img, 1, (200, 200, 200))
    # img_a = np.array(img)
    img = F.pad(img, 1, 200)

    # pad 3 to all sidess
    edge_padded_img = F.pad(img, 3, padding_mode='edge')
    # img_a = np.array(edge_padded_img)
    # First 6 elements of leftmost edge in the middle of the image, values are in order:
    # edge_pad, edge_pad, edge_pad, constant_pad, constant value added to leftmost edge, 0
    edge_middle_slice = np.asarray(edge_padded_img).transpose(2, 0, 1)[0][17][:6]
    # self.assertTrue(np.all(edge_middle_slice == np.asarray([200, 200, 200, 200, 1, 0])))
    # self.assertEqual(transforms.ToTensor()(edge_padded_img).size(), (3, 35, 35))
    assert np.all(edge_middle_slice == np.asarray([200, 200, 200, 200, 1, 0]))
    assert transforms.ToTensor()(edge_padded_img).size() == (3, 35, 35)

    # Pad 3 to left/right, 2 to top/bottom
    # reflect_padded_img = F.pad(img, (3, 2), padding_mode='reflect')
    reflect_padded_img = F.pad(img, (2, 3), padding_mode='reflect')
    # First 6 elements of leftmost edge in the middle of the image, values are in order:
    # reflect_pad, reflect_pad, reflect_pad, constant_pad, constant value added to leftmost edge, 0
    reflect_middle_slice = np.asarray(reflect_padded_img).transpose(2, 0, 1)[0][17][:6]
    # self.assertTrue(np.all(reflect_middle_slice == np.asarray([0, 0, 1, 200, 1, 0])))
    # self.assertEqual(transforms.ToTensor()(reflect_padded_img).size(), (3, 33, 35))
    assert np.all(reflect_middle_slice == np.asarray([0, 0, 1, 200, 1, 0]))
    assert transforms.ToTensor()(reflect_padded_img).size() == (3, 33, 35)

    # Pad 3 to left, 2 to top, 2 to right, 1 to bottom
    # symmetric_padded_img = F.pad(img, (3, 2, 2, 1), padding_mode='symmetric')
    symmetric_padded_img = F.pad(img, (2, 3, 1, 2), padding_mode='symmetric')
    # First 6 elements of leftmost edge in the middle of the image, values are in order:
    # sym_pad, sym_pad, sym_pad, constant_pad, constant value added to leftmost edge, 0
    symmetric_middle_slice = np.asarray(symmetric_padded_img).transpose(2, 0, 1)[0][17][:6]
    # self.assertTrue(np.all(symmetric_middle_slice == np.asarray([0, 1, 200, 200, 1, 0])))
    # self.assertEqual(transforms.ToTensor()(symmetric_padded_img).size(), (3, 32, 34))
    assert np.all(symmetric_middle_slice == np.asarray([0, 1, 200, 200, 1, 0]))
    assert transforms.ToTensor()(symmetric_padded_img).size() == (3, 32, 34)

    # Check negative padding explicitly for symmetric case, since it is not
    # implemented for tensor case to compare to
    # Crop 1 to left, pad 2 to top, pad 3 to right, crop 3 to bottom
    # symmetric_padded_img_neg = F.pad(img, (-1, 2, 3, -3), padding_mode='symmetric')
    symmetric_padded_img_neg = F.pad(img, (2, -1, -3, 3), padding_mode='symmetric')
    symmetric_neg_middle_left = np.asarray(symmetric_padded_img_neg).transpose(2, 0, 1)[0][17][:3]
    symmetric_neg_middle_right = np.asarray(symmetric_padded_img_neg).transpose(2, 0, 1)[0][17][-4:]
    # self.assertTrue(np.all(symmetric_neg_middle_left == np.asarray([1, 0, 0])))
    # self.assertTrue(np.all(symmetric_neg_middle_right == np.asarray([200, 200, 0, 0])))
    # self.assertEqual(transforms.ToTensor()(symmetric_padded_img_neg).size(), (3, 28, 31))
    assert np.all(symmetric_neg_middle_left == np.asarray([1, 0, 0]))
    assert np.all(symmetric_neg_middle_right == np.asarray([200, 200, 0, 0]))
    assert transforms.ToTensor()(symmetric_padded_img_neg).size() == (3, 28, 31)


def test_pad_raises_with_invalid_pad_sequence_len():
    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        transforms.Pad(())

    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        transforms.Pad((1, 2, 3))

    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        transforms.Pad((1, 2, 3, 4, 5))
