# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午10:21
@file: random_rotate.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.cls_transforms import RandomErasing, Compose, ToTensor


def main():
    img = imageio.imread('./assets/building.jpg')
    res_list = list()
    for i in range(1, 10):
        transform = Compose([
            ToTensor(),
            RandomErasing()
        ])
        res = transform(img)
        res_list.append(res.numpy().transpose(1, 2, 0))
    ia.show_grid(res_list)


if __name__ == '__main__':
    main()
