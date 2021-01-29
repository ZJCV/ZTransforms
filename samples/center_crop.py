# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:30
@file: center_crop.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.transforms import CenterCrop, Compose, ToPILImage, ToNumpyImage, ToTensor


def transform_pil(img, size):
    """
    :param img: ndarray
    """
    transform = Compose([
        ToPILImage(),
        CenterCrop(size),
        ToNumpyImage(),
    ])
    return transform(img)


def transform_tensor(img, size):
    transform = Compose([
        ToTensor(),
        CenterCrop(size),
        ToNumpyImage(),
    ])
    return transform(img)


def transform_numpy(img, size):
    transform = CenterCrop(size)
    return transform(img)


def main():
    func_dict = {
        'transform_pil': transform_pil,
        'transform_tensor': transform_tensor,
        'transform_numpy': transform_numpy
    }
    for name, func in func_dict.items():
        print(name)
        res_list = list()
        for i in range(1, 10):
            size = i * 40
            img = imageio.imread('./assets/building.jpg')
            res_list.append(func(img, size))
        ia.show_grid(res_list)


if __name__ == '__main__':
    main()
