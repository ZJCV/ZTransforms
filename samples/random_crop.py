# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:42
@file: random_crop.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms import RandomCrop


def main():
    img = imageio.imread('./assets/building.jpg')
    res_list = list()
    for i in range(1, 10):
        size = 200
        transform = RandomCrop(size)
        res = transform(img)
        res_list.append(res)
    ia.show_grid(res_list)


if __name__ == '__main__':
    main()
