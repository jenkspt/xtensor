import numpy as np
import cv2

import pytest
from xt import compress, decompress

def test_checkerboard_pattern():
    # Compress checkerboard patter
    im = np.zeros((4,6), dtype=np.float32)
    im[::2,::2] = 1
    im[1::2,1::2] = 1

    depth = 1
    c, a = compress(im, depth=depth, epsilon=0)
    im2 = decompress(c, a, depth=depth)

    assert np.all(im == im2)


def test_gradient_pattern():
    im = np.stack([np.arange(i,i+8) for i in range(8)], 0).astype(np.float32)

    depth = 2
    c, a = compress(im, depth=depth, epsilon=0)
    im2 = decompress(c, a, depth=depth)

    assert np.all(im == im2)

def test_cat_image():
    im = cv2.imread('cat.png', 0)
    depth = 7
    epsilon = 0

    c, a = compress(im.astype(np.float32), depth=depth, epsilon=epsilon)
    im2 = decompress(c, a, depth=depth).astype(np.uint8)
    assert np.all(im == im2)
