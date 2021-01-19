# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:49
@file: five_crop.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms import FiveCrop


def main():
    img = imageio.imread('./assets/building.jpg')

    transform = FiveCrop(200)
    res_list = transform(img)

    ia.show_grid(res_list, rows=1)


if __name__ == '__main__':
    main()
