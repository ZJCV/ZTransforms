# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:49
@file: five_crop.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.cls_transforms import TenCrop


def main():
    img = imageio.imread('./assets/building.jpg')

    transform = TenCrop(200)
    res_list = transform(img)

    ia.show_grid(res_list, rows=2)


if __name__ == '__main__':
    main()
