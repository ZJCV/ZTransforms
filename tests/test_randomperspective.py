# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午5:17
@file: test_randomperspective.py
@author: zj
@description: 
"""

import torch
import random
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F

import ztransforms.cls_transforms as transforms
import ztransforms.cls_functional as F


def test_randomperspective():
    for _ in range(10):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        img = torch.ones(3, height, width)
        # to_pil_image = transforms.ToPILImage()
        to_numpy_image = transforms.ToNumpyImage()
        # img = to_pil_image(img)
        img = to_numpy_image(img)
        perp = transforms.RandomPerspective()

        startpoints, endpoints = perp.get_params(width, height, 0.5)
        tr_img = F.perspective(img, startpoints, endpoints)
        tr_img2 = F.to_tensor(F.perspective(tr_img, endpoints, startpoints))
        tr_img = F.to_tensor(tr_img)

        # self.assertEqual(img.size[0], width)
        # self.assertEqual(img.size[1], height)
        # self.assertGreater(torch.nn.functional.mse_loss(tr_img, F.to_tensor(img)) + 0.3,
        #                    torch.nn.functional.mse_loss(tr_img2, F.to_tensor(img)))

        # assert img.size[0] == width
        # assert img.size[1] == height
        # assert (torch.nn.functional.mse_loss(tr_img, F.to_tensor(img)) + 0.3) > \
        #        torch.nn.functional.mse_loss(tr_img2, F.to_tensor(img))

        assert img.shape[0] == height
        assert img.shape[1] == width
        assert (torch.nn.functional.mse_loss(tr_img, F.to_tensor(img)) + 0.3) > \
               torch.nn.functional.mse_loss(tr_img2, F.to_tensor(img))
