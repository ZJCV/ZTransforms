# -*- coding: utf-8 -*-

"""
@date: 2021/1/13 下午5:14
@file: perspective.py
@author: zj
@description: 
"""

import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms

import imageio
import imgaug as ia
import imgaug.augmenters as iaa


# aug = iaa.PerspectiveTransform(scale=0.5, keep_size=False)
# print(aug)
#
# out = aug.augment_image(img)
# ia.imshow(out)

# img = Image.open('./assets/lena.jpg')
# aug2 = transforms.RandomPerspective()
# out = aug2(img)
# out.show()


def get_params(width, height, distortion_scale):
    """Get parameters for ``perspective`` for a random perspective transform.

    Args:
        width : width of the image.
        height : height of the image.

    Returns:
        List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
    """
    half_height = int(height / 2)
    half_width = int(width / 2)
    topleft = (random.randint(0, int(distortion_scale * half_width)),
               random.randint(0, int(distortion_scale * half_height)))
    topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                random.randint(0, int(distortion_scale * half_height)))
    botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
    botleft = (random.randint(0, int(distortion_scale * half_width)),
               random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
    startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    endpoints = [topleft, topright, botright, botleft]
    return startpoints, endpoints


def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the orignal image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image,
        List containing [top-left, top-right, bottom-right, bottom-left] of the transformed
                   image
    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    return cv2.getPerspectiveTransform(np.float32(startpoints), np.float32(endpoints))


def perspective(img, startpoints, endpoints, interpolation=cv2.INTER_LINEAR, fill=None):
    """Perform perspective transform of the given Numpy Image.

    Args:
        img (Numpy Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- cv2.INTER_LINEAR
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            This option is only available for ``pillow>=5.0.0``.

    Returns:
        Numpy Image:  Perspectively transformed Image.
    """
    M = _get_perspective_coeffs(startpoints, endpoints)
    return cv2.warpPerspective(img, M, img.shape[:2], flags=interpolation, borderValue=fill)


img = imageio.imread('./assets/lena.jpg')
# ia.imshow(img)


h, w = img.shape[:2]
pt1, pt2 = get_params(h, w, 0.5)
print(pt1, pt2)

print(_get_perspective_coeffs(pt1, pt2))

M = cv2.getPerspectiveTransform(np.float32(pt1), np.float32(pt2))
print(M)

dst = perspective(img, pt1, pt2)
ia.imshow(dst)

# dst = cv2.warpPerspective(img, M, img.shape[:2])
# ia.imshow(dst)
