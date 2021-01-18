# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午7:31
@file: random_perspective.py
@author: zj
@description: 
"""

import imgaug as ia
import imageio

import ztransforms.cls_transforms as transforms
import ztransforms.cls_functional as F


def random_perspective():
    img = imageio.imread('./assets/lena.jpg')
    height, width = img.shape[:2]
    perp = transforms.RandomPerspective()

    startpoints, endpoints = perp.get_params(width, height, 0.5)
    print(startpoints, endpoints)
    tr_img = F.perspective(img, startpoints, endpoints)
    tr_img2 = F.perspective(tr_img, endpoints, startpoints)

    ia.imshow(img)
    ia.imshow(tr_img)
    ia.imshow(tr_img2)


if __name__ == '__main__':
    random_perspective()
