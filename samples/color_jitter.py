# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:54
@file: color_jitter.py
@author: zj
@description: 
"""

import numpy as np
import imageio
import imgaug as ia
from ztransforms import ColorJitter


def main():
    img = imageio.imread('./assets/building.jpg')
    res_list = list()
    for i in range(1, 10):
        brightness = np.random.randint(1, 5)
        contrast = np.random.randint(1, 5)
        saturation = np.random.randint(1, 5)
        hue = np.random.randint(-255, 255)
        transform = ColorJitter(brightness, contrast, saturation, hue)
        res = transform(img)
        res_list.append(res)
    ia.show_grid(res_list)


if __name__ == '__main__':
    main()
