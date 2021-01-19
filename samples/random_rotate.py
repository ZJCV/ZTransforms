# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午10:21
@file: random_rotate.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms import RandomRotation


def main():
    img = imageio.imread('./assets/building.jpg')
    res_list = list()
    for i in range(1, 10):
        transform = RandomRotation(100)
        res = transform(img)
        res_list.append(res)
    ia.show_grid(res_list)


if __name__ == '__main__':
    main()
