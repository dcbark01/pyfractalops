import time
from PIL import Image
from typing import Tuple, List, Union

import numpy as np


def load_image(img_file: str, mode='L'):
    """ Open an image using PIL (defaults to grayscale). """
    return Image.open(img_file).convert(mode)


def pil2array(pil_img):
    return np.array(pil_img)


def array2pil(img_arr):
    return Image.fromarray(img_arr.astype(np.uint8))


def resize_image(img, size: (int, tuple, list)):
    """ Resize an image (either PIL or numpy array acceptable, same type is returned).
    Note that PIL uses (W x H), but I always use convention (H x W), so we need to swap here
    """
    if isinstance(size, int):
        h, w = (size, size)
    elif isinstance(size, (tuple, list)):
        h, w = size

    if isinstance(img, np.ndarray):
        img_type = 'numpy'
        img_r = array2pil(img)
    elif isinstance(img, Image.Image):
        img_type = 'pil'
        img_r = img
    else:
        raise ValueError('Resize value must be either an int or a 2-tuple/list of (H x W)!!!')

    img_r = img_r.resize((w, h))

    # Convert blurred image back to original type
    if img_type == 'numpy':
        return pil2array(img_r)
    else:
        return img_r


def crop_image(img: np.ndarray, crop_size: Union[int, List, Tuple]):
    if isinstance(crop_size, int):
        wr, hr = (crop_size, crop_size)
    elif isinstance(crop_size, (list, tuple)):
        hr, wr = crop_size

    h, w = img.shape[0:2]

    crop_h = h - hr
    crop_top = np.abs(crop_h) // 2
    crop_bottom = np.abs(crop_h) - crop_top

    crop_w = w - wr
    crop_left = np.abs(crop_w) // 2
    crop_right = np.abs(crop_w) - crop_left

    img_crop = img[crop_left:(w - crop_right), crop_top:(h - crop_bottom), ...]
    return img_crop


def get_num_image_channels(img_arr: np.ndarray):
    if len(img_arr.shape) == 2:
        c = 1
    elif len(img_arr.shape) == 3:
        c = img_arr.shape[2]
        if c != 3:
            raise ValueError('Image to pad must be grayscale array of shape (H x W) or RGB of (H x W x 3)!')
    return c


def convolve2d(arr, kernel, pad_mode='constant', pad_value=255):
    """ Convolve the kernel with the 2d input array. """
    # Need to handle padding of input/output to guarantee output has same shape as input
    pl = int(np.ceil((kernel.shape[0] - 1) / 2))
    pr = int(np.floor((kernel.shape[0] - 1) / 2))
    pt = int(np.ceil((kernel.shape[1] - 1) / 2))
    pb = int(np.floor((kernel.shape[1] - 1) / 2))
    # print(pl, pr, pt, pb)
    arr_pad = np.pad(arr, ((pt, pb), (pl, pr)), mode=pad_mode, constant_values=pad_value)
    output_shape = kernel.shape + tuple(np.subtract(arr_pad.shape, kernel.shape) + 1)

    submatrices = np.lib.stride_tricks.as_strided(arr_pad, shape=output_shape, strides=arr_pad.strides * 2)
    arr_conv = np.einsum('ij,ijkl->kl', kernel, submatrices)
    return arr_conv


def union2d(img1, img2, threshold=128):
    img1 = np.where(img1 < threshold, 1, 0)
    img2 = np.where(img2 < threshold, 1, 0)
    img_union = np.where((img1 == 1) | (img2 == 1), 0, 255)
    return img_union


def subtraction2d(img1, img2):
    img1 = np.where(img1 > 128, 0, 255)
    img2 = np.where(img2 > 128, 0, 255)
    img_sub = img1 - img2
    # img_sub = np.clip(img_sub, 0, 255)
    # img_sub = blur_image(img_sub, kernel_size=5, sigma=1.5)
    img_sub = np.where(img_sub > 128, 0, 255)
    return img_sub


def intersection2d(img1, img2, threshold=128):

    img1 = np.where(img1 < threshold, 1, 0)
    img2 = np.where(img2 < threshold, 1, 0)
    img_intersect = np.where((img1 * img2 > 0), 0, 255)
    return img_intersect


def xor2d(img1, img2):
    """ Get logical XOR between two images (using blurring to help eliminate stray image artifacts). """
    img_xor = np.logical_xor(img1, img2)
    img_xor = np.where(img_xor == True, 255, 0)
    img_xor_smooth = blur_image(img_xor, kernel_size=5, sigma=1.0)
    # img_xor_smooth = np.where(img_xor_smooth > 90, 255, 0)

    # Invert image so that XORed objects show up as black
    img_xor_smooth = np.where(img_xor_smooth > 90, 0, 255)   # TODO: Does this make since when using, eg MSE error?
    return img_xor_smooth


def jaccard_image_similarity(img1, img2):
    # See Equation 1 in Kunda, 2010
    s = intersection2d(img1, img2).sum() / union2d(img1, img2).sum()
    return s


def tversky_image_similarity(img1, img2, alpha=1.0, beta=1.0):
    # See Equation 1 in Kunda, 2010
    img1_bin = np.where(img1 < 128, 1, 0).astype(bool)
    img2_bin = np.where(img2 < 128, 1, 0).astype(bool)
    img_intersect = np.where((img1_bin * img2_bin > 0), 0, 1)

    img_union = np.where(img1_bin | img2_bin, 0, 1)

    s = img_intersect.sum() / (img_union.sum() + 1e-6)
    return s


def rmse(a, b):
    """ Root-mean-square error calculation for two arrays. """
    return np.sqrt(np.sum((a - b)**2))


def sobel_edge_detector(img, L=255.0):
    kh = np.array([
        [-1, -2, 0, 2, 1],
        [-4, -8, 0, 8, 4],
        [-6, -12, 0, 12, 6],
        [-4, -8, 0, 8, 4],
        [-1, -2, 0, 2, 1]], dtype=np.float)

    kv = np.array([
        [1, 4, 6, 4, 1],
        [2, 8, 12, 8, 2],
        [0, 0, 0, 0, 0],
        [-2, -8, -12, -8, -2],
        [-1, -4, -6, -4, -1]], dtype=np.float)

    gx = convolve2d(img, kh)
    gy = convolve2d(img, kv)

    g = np.sqrt(gx ** 2 + gy ** 2)

    eps = 1e-6   # Avoid division by zero issues
    g = (g * L) / (np.max(g) + eps)

    # Sobel filter will return edges as white and background as black,
    # So we'll need to invert the colors
    g = 255 - g
    return g


def gaussian_kernel_2d(size, sigma):
    """Returns a 2D square Gaussian kernel of shape (size by size)
    Source: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    x = np.linspace(- (size // 2), size // 2, size)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()


def blur_image(img_arr: np.ndarray, kernel_size=3, sigma=3.0):
    """ Blur an image using a Gaussian kernel specified by 'size' arg.
    Image must be grayscale array of shape (H x W) or RGB of (H x W x 3)
    """
    c = get_num_image_channels(img_arr)
    gk = gaussian_kernel_2d(kernel_size, sigma=sigma)
    img_b = np.empty_like(img_arr)
    if c == 1:
        img_b[:, :] = convolve2d(img_arr, gk)
    elif c == 3:
        for i in range(c):
            img_b[:, :, i] = convolve2d(img_arr[:, :, i], gk)
    return img_b


def timeit(method):
    """ Decorator function for getting execution time of a function.
    Source: Unknown, I use it frequently for different projects so I copied from some old code I wrote, but I know
            I did not originally write it. I believe I initially found it on StackExchange.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def show_image(img, ax=None, **kwargs):
    """ Compares to grayscale numpy array images in a side-by-side plot. """
    try:
        import matplotlib.pyplot as plt

        title = kwargs.get('title', '')
        vmin = kwargs.get('vmin', 0)
        vmax = kwargs.get('vmax', 255)
        cmap = kwargs.get('cmap', 'gray')

        if not ax:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(title)

        if not ax:
            plt.show()
            plt.close()
        else:
            return ax
    except ImportError as e:
        print('Matplotlib not available in current environment. Skipping image transform preview.')
