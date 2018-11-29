import numpy as np
from scipy.ndimage import zoom
import scipy.sparse as sparse
import cv2

import matplotlib.pyplot as plt

def compress(arr, depth=1, epsilon=0):
    """
    Args:
        arr     (ndarray): 2d array
        depth      (int) : Positive integer. Number of compression
        iterations. 2^depth must be less than the min dimention of arr
        epsilon (float) : Threshold below which errors are set to zero
    """
    order=0     # Use nearest neighbors interpolation

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


def decompress(c, anchor, depth=1):
    """
    Args:
        arr     (ndarray):
        depth      (int) : Positive integer.
        epsilon (number) :
            Should be a valid value in arr.dtype
    """
    order = 0

    for i in reversed(range(depth)):
        s = 2**i
        error = c[::s,::s]
        n0,n1 = error.shape

        interp = zoom(anchor, 2, order=order)[:n0,:n1]
        anchor = error + interp
        error[...] = 0

    return anchor

if __name__ == "__main__":
    im = cv2.imread('cat.png', 0)
    plt.imshow(im, cmap='gray'); plt.show()


    depth = 7
    epsilon = 0

    c, a = compress(im.astype(np.float32), depth=depth, epsilon=epsilon)

    plt.imshow(c, cmap='gray'); plt.show()
    # Save compressed image
    plt.imsave(f'c{epsilon}.png', 
            np.interp(c, [c.min(), c.max()],[0, 255]).round(0).astype(np.uint8),
            cmap='gray')

    dok = sparse.dok_matrix(c.copy())
    print(f'Percent Non Zero: {dok.nnz/np.product(dok.shape) * 100:.2f}%')
    im2 = decompress(c, a, depth=depth).astype(np.uint8)

    plt.imshow(im2, cmap='gray'); plt.show()
    plt.imsave(f'dc{epsilon}.png', im2, cmap='gray')
