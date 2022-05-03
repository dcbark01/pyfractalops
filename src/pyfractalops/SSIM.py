from collections import namedtuple
from functools import lru_cache, wraps

import numpy as np

from pyfractalops.utils import blur_image, crop_image


def calculate_ssim(img1, img2, window_size=11, sigma=1.0, L=255):
    """ Compute the Structural Similarity Index (SSIM) for two input images.

    For reference, see:
       https://www.mathworks.com/help/images/ref/ssim.html
       https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf   (Original 2004 paper by Wang, et al)

    :param img1:
    :param img2:
    :param L:      (int) L is the specified DynamicRange value.
    :return:       (tuple) Mean SSIM value for images (float in range [-1, 1] as well as pixelwise array
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * L)**2,
    C2 = (0.03 * L)**2

    # Calculate local means, standard deviations, and cross-covariance for images
    # Means
    U_x = blur_image(img1, kernel_size=3, sigma=sigma)
    U_y = blur_image(img2, kernel_size=3, sigma=sigma)

    # Standard deviations
    S_xx = blur_image(img1 * img1, kernel_size=3, sigma=sigma)
    S_yy = blur_image(img2 * img2, kernel_size=3, sigma=sigma)
    S_xy = blur_image(img1 * img2, kernel_size=3, sigma=sigma)

    # Covariances
    cov_x = S_xx - (U_x * U_x)
    cov_y = S_yy - (U_y * U_y)
    cov_xy = S_xy - (U_x * U_y)

    # Equation 13 (Wang, et al, 2004)
    ssim = ((2 * U_x * U_y + C1) * (2 * cov_xy + C2)) / ((U_x**2 + U_y**2 + C1) * (cov_x + cov_y + C2))

    # Ignore calculating SSIM for image border regions where full sliding window couldn't be applied
    border = (window_size - 1) // 2
    mu_ssim = crop_image(ssim, border).mean()
    return mu_ssim, ssim


def psi_mexh(width, num_points):
    """ Create a Mexican hat wavelet (aka Ricker wavelet) of the specified size.
    See Wikipedia for definition:
       https://en.wikipedia.org/wiki/Mexican_hat_wavelet
    """
    t = np.arange(0, num_points) - (num_points - 1.0) / 2
    psi = (2 / (np.sqrt(3 * width) * np.pi ** 0.25)) * (1 - (t / width) ** 2) * np.exp(-(t ** 2 / (2 * width ** 2)))
    return psi


def cwt_mexh(x, widths):
    """ Calculate the complex wavelet transform for the input array. """
    sx = np.zeros((len(widths), len(x)))
    for level, width in enumerate(widths):
        N = np.min([10 * width, len(x)])

        wav_kernel = np.conj(psi_mexh(width, num_points=N)[::-1])

        # Calculate padding for convolution
        pl = int(np.ceil((len(wav_kernel) - 1) / 2))
        pr = int(np.floor((len(wav_kernel) - 1) / 2))
        x_pad = np.pad(x, (pl, pr))

        xwc = np.convolve(x_pad, wav_kernel)[pl:len(x_pad) - pr]
        sx[level] = xwc
    return sx


def cw_ssim(img1, img2, width=30, K=0.01):
    """ Calculate the complex wavelet structural similarity index (CW-SSIM).
    See Gao, et al. (2011) "CW-SSIM based image classification"
    """

    # Collapse images into 1-D arrays and calculate wavelet widths to use
    # Also, assume images are same shape
    h, w = img1.shape[0:2]
    s1 = img1.flatten()
    s2 = img2.flatten()
    widths = np.arange(1, width + 1)

    swav1 = cwt_mexh(s1, widths)
    swav2 = cwt_mexh(s2, widths)

    c1c2 = np.abs(swav1) * np.abs(swav2)
    c1c2_conj = swav1 * np.conjugate(swav2)
    c1_squared = np.abs(swav1) ** 2
    c2_squared = np.abs(swav2) ** 2
    num11 = 2 * np.sum(c1c2, axis=0) + K
    den11 = np.sum(c1_squared, axis=0) + np.sum(c2_squared, axis=0) + K
    num22 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + K
    den22 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + K

    ssim = (num11 / den11) * (num22 / den22)
    img_ssim = ssim.reshape((h, w))
    mu_ssim = img_ssim.mean()
    return mu_ssim, img_ssim


CachedCWSSIMArgs = namedtuple('CachedCWSSIMArgs', ['img1', 'img2', 'width'])


def numpy_cw_ssim_cache_map(*args, **kwargs):
    """ LRU cache implementation for functions whose FIRST parameter is a numpy array

    Modified from source:
        https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
    """

    def decorator(function):
        @wraps(function)
        def wrapper(args_tuple: CachedCWSSIMArgs):
            args_tuple_hashable = CachedCWSSIMArgs(
                img1=tuple(map(tuple, args_tuple.img1)),
                img2=tuple(map(tuple, args_tuple.img2)),
                width=args_tuple.width
            )
            return cached_wrapper(args_tuple_hashable, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(args_tuple_hashable: CachedCWSSIMArgs, *args, **kwargs):
            args_tuple = CachedCWSSIMArgs(
                img1=np.array(args_tuple_hashable.img1),
                img2=np.array(args_tuple_hashable.img2),
                width=args_tuple_hashable.width
            )
            return function(args_tuple, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator


@numpy_cw_ssim_cache_map()
def cw_ssim_cached(args_tuple: CachedCWSSIMArgs):
    """ Perform the fractal encoding of the range image in terms of the given domain image. """
    mu_ssim, ssim_img = cw_ssim(
        args_tuple.img1,
        args_tuple.img2,
        args_tuple.width
    )
    return mu_ssim, ssim_img
