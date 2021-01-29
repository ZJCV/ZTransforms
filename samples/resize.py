# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:38
@file: resize.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.transforms import Compose, Resize, ToPILImage, ToTensor, ToNumpyImage


def resize_pil(img, size):
    """
    :param img: ndarray
    """
    transform = Compose([
        ToPILImage(),
        Resize(size),
        ToNumpyImage(),
    ])
    return transform(img)


def resize_tensor(img, size):
    transform = Compose([
        ToTensor(),
        Resize(size),
        ToNumpyImage(),
    ])
    return transform(img)


def resize_numpy(img, size):
    transform = Resize(size)
    return transform(img)


def main():
    func_dict = {
        'resize_pil': resize_pil,
        'resize_tensor': resize_tensor,
        'resize_numpy': resize_numpy
    }
    for name, func in func_dict.items():
        print(name)
        res_list = list()
        for i in range(1, 10):
            size = i * 80
            img = imageio.imread('./assets/building.jpg')
            res_list.append(func(img, size))
        ia.show_grid(res_list)


if __name__ == '__main__':
    main()
