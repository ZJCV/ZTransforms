# -*- coding: utf-8 -*-

"""
@date: 2021/1/12 下午2:41
@file: load_image.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(4)

ia.imresize_single_image()
ia.imresize_many_images()

iaa.CenterCropToSquare

def read_img():
    img = imageio.imread('../imgs/lena.jpg')
    ia.imshow(img)

    return img


def rotate(img):
    rotate = iaa.Affine(rotate=(-50, 50))
    image_aug = rotate(image=img)

    print("Augmented:")
    ia.imshow(image_aug)

    import numpy as np

    images = [img, img, img, img]
    images_aug = rotate(images=images)

    print("Augmented batch:")
    ia.imshow(np.hstack(images_aug))


if __name__ == '__main__':
    img = read_img()
    rotate(img)
    # batch(img)
