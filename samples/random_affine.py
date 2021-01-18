# -*- coding: utf-8 -*-

"""
@date: 2021/1/14 上午9:39
@file: random_affine.py
@author: zj
@description: 
"""

from PIL import Image
import math
from math import sin, cos, tan
import numbers
import random
import numpy as np

import imageio
import imgaug as ia
import imgaug.augmenters as iaa


def get_params(degrees, translate, scale_ranges, shears, img_size):
    """Get parameters for affine transformation

    Returns:
        sequence: params to be passed to the affine transformation
    """
    angle = random.uniform(degrees[0], degrees[1])
    if translate is not None:
        max_dx = translate[0] * img_size[0]
        max_dy = translate[1] * img_size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        if len(shears) == 2:
            shear = [random.uniform(shears[0], shears[1]), 0.]
        elif len(shears) == 4:
            shear = [random.uniform(shears[0], shears[1]),
                     random.uniform(shears[2], shears[3])]
    else:
        shear = 0.0

    return angle, translations, scale, shear


def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (PIL Image): PIL Image to be rotated.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
        If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
        the second value corresponds to a shear parallel to the y axis.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter.
            See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
        "Argument translate should be a list or tuple of length 2"

    assert scale > 0.0, "Argument scale should be positive"

    output_size = img.size
    center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
    matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
    kwargs = {"fillcolor": fillcolor}
    return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


def _get_inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    if isinstance(shear, numbers.Number):
        shear = [shear, 0]

    if not isinstance(shear, (tuple, list)) and len(shear) == 2:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " +
            "two values. Got {}".format(shear))

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = cos(rot - sy) / cos(sy)
    b = -cos(rot - sy) * tan(sx) / cos(sy) - sin(rot)
    c = sin(rot - sy) / cos(sy)
    d = -sin(rot - sy) * tan(sx) / cos(sy) + cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    M = [d, -b, 0,
         -c, a, 0]
    M = [x / scale for x in M]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
    M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    M[2] += cx
    M[5] += cy
    return M


degrees = (-35, 60)
translate = (0.2, 0.4)
scale_ranges = (0.5, 1)
shears = (-5, 5)
print(degrees, translate, scale_ranges, shears)

img = imageio.imread('./assets/lena.jpg')
ia.imshow(img)

img_size = img.shape[:2]
angle, translations, scale, shear = get_params(degrees, translate, scale_ranges, shears, img_size)
print(angle, translations, scale, shear)
aug = iaa.Affine(scale=scale, rotate=angle, translate_px=(int(translations[0]), int(translations[1])), shear=shear)
res_img = aug.augment_image(img)
ia.imshow(res_img)

img = Image.open('assets/lena.jpg')
# res_img = affine(img, angle, translations, scale, shear)
# img.show()
# res_img.show()

center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
print(matrix)