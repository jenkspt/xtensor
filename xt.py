import numpy as np
from scipy.ndimage import map_coordinates
import scipy.sparse as sparse
from PIL import Image

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

    def to_pil_image(x, dtype=np.uint8):
        scaled = np.interp(x, [x.min(), x.max()], [0,255])
        scaled = np.clip(scaled.round(0),0,255).astype(dtype)
        return Image.fromarray(scaled)

    im = Image.open('images/cat-gray.png')
    
    depth = 7

    # Scale image to reduce floating point errors
    input = np.array(im, dtype=np.float32)/255

    for order in (0,1,3):
        for epsilon in (0, 2.5, 10, 25):

            c, a = compress(input, depth=depth, order=order, epsilon=epsilon/255)
            pnz = len(c.nonzero()[0])/c.size * 100    # Percent non-zero
            cim = to_pil_image(c).save(f'images/c_eps-{epsilon}_ord-{order}.png')

            #dok = sparse.dok_matrix(c.copy())
            dc = decompress(c, a, depth=depth, order=order)
            dcim = to_pil_image(dc).save(f'images/dc_eps-{epsilon}_ord-{order}.png')

            error = dc - input
            perr = np.abs(error*255).round(0).sum()/error.size
            dcim = to_pil_image(error).save(f'images/error_eps-{epsilon}_ord-{order}.png')
            print(f'Epsilon: {epsilon}, order: {order}, non-zero: {pnz:.1f}%, error {perr:.1f}%')

            if epsilon == 0:
                assert np.allclose(input, dc, atol=1e-6)
