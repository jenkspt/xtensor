import numpy as np
from scipy.ndimage import map_coordinates
import scipy.sparse as sparse
import cv2

import matplotlib.pyplot as plt

def zoom(input, factor, order=0, mode='nearest'):
    indices = np.indices(np.array(input.shape)*factor)/factor
    return map_coordinates(input, indices, order=order, mode=mode)


def compress(arr, depth=1, order=0, epsilon=0):
    """
    Args:
        arr     (ndarray): 2d array
        depth      (int) : Positive integer. Number of compression
        order      (int) : Spline interpolation order
        iterations. 2^depth must be less than the min dimention of arr
        epsilon (float) : Threshold below which errors are set to zero
    """

    compressed = np.zeros_like(arr)
    for i in range(depth):
        s = 2**i    # Stride
        # View image data with the given stride
        anchor = arr[::s, ::s]
        n0, n1 = anchor.shape

        interp = zoom(anchor[::2,::2], 2, order=order)[:n0,:n1]
        error = anchor - interp

        mask = np.abs(error) > epsilon
        # Save error values that exceed epsilon
        compressed[::s,::s][mask] = error[mask]
    #return compressed, anchor[::2,::2]
    return compressed, anchor[::2,::2]


def decompress(c, anchor, depth=1, order=0):
    """
    Args:
        arr     (ndarray):
        depth      (int) : Positive integer.
        order      (int) : Spline interpolation order
        epsilon (number) :
            Should be a valid value in arr.dtype
    """

    for i in reversed(range(depth)):
        s = 2**i
        error = c[::s,::s]
        n0,n1 = error.shape

        interp = zoom(anchor, 2, order=order)[:n0,:n1]
        anchor = error + interp
        error[...] = 0

    return anchor

if __name__ == "__main__":
    im = cv2.imread('images/cat.png', 0)
    plt.imshow(im, cmap='gray'); plt.show()


    depth = 7
    epsilon = 10
    order = 0

    c, a = compress(im.astype(np.float32), depth=depth, order=order, epsilon=epsilon)

    cimg = np.interp(c, [-255,255],[0,1])
    plt.imshow(cimg, cmap='gray'); plt.show()
    # Save compressed image
    plt.imsave(f'images/c{epsilon}.png', 
            np.interp(c, [c.min(), c.max()],[0, 255]).round(0).astype(np.uint8),
            cmap='gray')

    dok = sparse.dok_matrix(c.copy())
    print(f'Percent Non Zero: {dok.nnz/np.product(dok.shape) * 100:.2f}%')
    im2 = decompress(c, a, depth=depth, order=order)

    if epsilon == 0:
        assert np.allclose(im, im2, atol=1e-6)

    plt.imshow(im2, cmap='gray'); plt.show()
    plt.imsave(f'images/dc{epsilon}.png', im2, cmap='gray')
