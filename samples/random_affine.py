# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午10:23
@file: random_affine.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms import RandomAffine


def main():
    img = imageio.imread('./assets/building.jpg')
    res_list = list()
    for i in range(1, 10):
        degrees = 40
        translate = (0.4, 0.7)
        scale = (0.5, 2.)
        shear = 2
        transform = RandomAffine(degrees, translate, scale, shear)
        res = transform(img)
        res_list.append(res)
    ia.show_grid(res_list)


if __name__ == '__main__':
    main()
