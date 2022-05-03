import numpy as np
import multiprocessing
from multiprocessing import Pool
from contextlib import contextmanager
from typing import Tuple, List, Union
from functools import partial, lru_cache, wraps
from collections import namedtuple

from pyfractalops.utils import resize_image
from pyfractalops.RavensProblem import RavensProblem
from pyfractalops.Transforms import create_candidate_transforms
from pyfractalops.ImageComposition import find_best_image_composition, apply_composition
from pyfractalops.FractalFeatures import features_from_fractals, feature_union, tversky_feature_similarity


SIM_RMSE_DISCARD_THRESHOLD = 5.0
TRANSFORMS = create_candidate_transforms(max_n=1, max_complexity=10)
MIN_BLACK_PIXELS = 1


def divide_chunks(my_list: Union[List, Tuple], n: int):
    """ Helper function for breaking lists into bite-size chunks of size n.
    Source: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    :returns Generator
    """
    # Looping till length l
    for i in range(0, len(my_list), n):
        yield my_list[i:i + n]


def divide_chunks_2d(my_list: Union[List, Tuple], chunk_size: int):
    """ Helper function for breaking lists into bite-size chunks of size n.
    Modified from source: https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    :returns Generator
    """
    # Looping till length l
    for chunk in range(chunk_size):
        for i in range(len(my_list)):
            for j in range(len(my_list[0])):
                patch = my_list[i][j]
                yield patch
                # yield my_list[i:i + n]


def get_cpu_count():
    return multiprocessing.cpu_count()


def set_num_process_workers(cpus):
    """ Sets the number of CPUs based on user inputs.
    If cpus == None, then all available cores are used (found by inspecting system resources using multiprocessing lib)
    """
    cpus = cpus if cpus is not None else get_cpu_count()
    assert isinstance(cpus, int) and cpus > 0, "Number of CPUs to use must be int > 0"
    # TODO: Will the autograder let me check to see how many worker processes I can use????
    if cpus > get_cpu_count():
        cpus = get_cpu_count()    # // 2  # Only use half of the cpus available
    return cpus


@contextmanager
def poolcontext(*args, **kwargs):
    """ Create a context manager for handling worker processes.
    See StackOverflow answer here:
        https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
    """
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def reduce_image(img, scale_factor):
    """ Downsample the image by the specified factor by taking the mean"""
    h_orig, w_orig = img.shape[0:2]
    img_red = np.zeros((h_orig // scale_factor, w_orig // scale_factor))
    h_new, w_new = img_red.shape[0:2]
    for i in range(h_new):
        for j in range(w_new):
            img_red[i, j] = np.mean(img[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor])
    return img_red


def calculate_whitespace_area(img):
    """ Returns the number of white pixels in the input image. """
    h, w = img.shape[0:2]
    img_bin = np.where(img > 10, 0, 1)   # Binarize and invert so black pixels := 1, white := 0
    area_black = np.sum(img_bin)
    area_white = (h * w) - area_black
    return area_white


def calculate_whitespace_ratio(img):
    """ Return the ratio of white pixels (ie background) to total image area (1.0 indicates all whitespace). """
    area_white = calculate_whitespace_area(img)
    h, w = img.shape[0:2]
    white_ratio = area_white / (h * w)
    return white_ratio


def is_not_whitespace(img):
    img_bin_black = np.where(img > 128, 0, 1).astype(bool)
    if np.sum(img_bin_black) >= MIN_BLACK_PIXELS:
        return True
    else:
        return False


def rmse(a, b):
    """ Root mean square error calculation for two arrays. """
    return np.sqrt(np.sum((a - b)**2))


def _color_contraction(b_i, a_k):
    """ Calculate color contraction per McGreggor 2014 section 3.4.3.9
    Equivalent names in paper
    D  ::  a_k
    R  ::  b_i
    colorContraction ( a k , b i ) = 0 . 75 ∗ (colorMean ( b i ) − colorMean ( a k ))
    """
    contrast = 0.75
    brightness = int(np.floor((np.sum(b_i - contrast * a_k)) / b_i.size))
    brightness = int(np.clip(brightness, -255, 255))
    return contrast, brightness


def _get_transformed_domain_partitions_single_image(img, grid_size):

    step_size = grid_size
    h, w = img.shape[0:2]

    k_coords = [k for k in range(((h - grid_size) // step_size) + 1)]
    l_coords = [l for l in range(((w - grid_size) // step_size) + 1)]
    partitions_transformed = []

    for k in k_coords:
        for l in l_coords:
            sx = k * step_size
            sy = l * step_size
            source_fragment = img[sx:sx + grid_size, sy:sy + grid_size]

            if is_not_whitespace(source_fragment):
                for ti, T in enumerate(TRANSFORMS):
                    D = T(source_fragment)
                    partitions_transformed.append((sx, sy, ti, D))

    return partitions_transformed


def _get_transformed_domain_partitions_image_pair(img1, img2, grid_size):

    assert img1.shape == img2.shape, "Domain partitioning for pair of images requires images of same shape!"

    step_size = grid_size
    h, w = img1.shape[0:2]

    k_coords = [k for k in range(((h - grid_size) // step_size) + 1)]
    l_coords = [l for l in range(((w - grid_size) // step_size) + 1)]
    partitions_transformed = []

    for k in k_coords:
        for l in l_coords:
            sx = k * step_size
            sy = l * step_size
            source_fragment1 = img1[sx:sx + grid_size, sy:sy + grid_size]
            source_fragment2 = img2[sx:sx + grid_size, sy:sy + grid_size]

            if any([is_not_whitespace(source_fragment1), is_not_whitespace(source_fragment2)]):
                for ti, T in enumerate(TRANSFORMS):
                    D1 = T(source_fragment1)
                    D2 = T(source_fragment2)
                    partitions_transformed.append((sx, sy, ti, D1, D2))

    return partitions_transformed


def _get_transformed_domain_partitions(images: Union[Tuple, List], grid_size: int) -> List:
    """ Wrapper for selecting using smart or dumb domain transform generator. """
    if len(images) == 1:
        img = images[0]
        return _get_transformed_domain_partitions_single_image(img, grid_size, ignore_whitespace=True)
    else:
        assert len(images) == 2, 'Can only partition a tuple/list containing a single image or two images!'
        img1, img2 = images
        return _get_transformed_domain_partitions_image_pair(img1, img2, grid_size, ignore_whitespace=True)


def _create_blank_image_codebook(range_partitions):
    """" This is to help handle cases like Basic Problem C-08 Image A where it is all blank"""
    codebook = []
    for dx, dy, _ in range_partitions:
        C = 0.75    # Default contrast
        B = 255     # All white
        ti = 0
        codebook.append([dx, dy, 0, 0, ti, C, B])
    return codebook


def _get_range_image_partitions(range_image, range_grid_size):
    i_coords = range_image.shape[0] // range_grid_size
    j_coords = range_image.shape[1] // range_grid_size
    range_partitions = []
    for i in range(i_coords):
        for j in range(j_coords):
            dx = i * range_grid_size
            dy = j * range_grid_size
            R = range_image[dx:dx + range_grid_size, dy:dy + range_grid_size]
            range_partitions.append((dx, dy, R))
    return range_partitions


def _calculate_correspondence_weights(transformed_domain_partitions, range_partitions,
                                      minimum_noticeable_photometric=1):
    """ See 3.4.11 'Refining Correspondence' of McGreggor (2014). """
    sxx = min(set([x[0] for x in transformed_domain_partitions]))
    syy = min(set([y[1] for y in transformed_domain_partitions]))
    dxx = max(set([x[0] for x in range_partitions]))
    dyy = max(set([y[1] for y in range_partitions]))

    max_d = _fragment_distance(dxx, dyy, sxx, syy)

    """
    From paper, we have: w2 * maximalDistance << w1 * minimalJustNoticeablePhotometric
    Distance and min noticable photometric will never be negative, so we can solve for w2:
       Let w1 = minimumJustNoticablePhotometric == 1   (grayscale)
       w2 = minJustNoticeablePhotometric) / maximal_distance

    where 'eps' is some positive non-zero number to prevent division by zero issue in the case where
    our grid size is equal to the full dimensions of the image, for example, in the case where our image size is 
    128x128 and our grid size is also 128, this would give us a max distance of zero, so we need to ensure
    w2 doesn't go to nan
    """
    eps = 10
    w1 = minimum_noticeable_photometric
    w2 = w1 / (max_d + eps)
    return w1, w2


def _fragment_distance(dx, dy, sx, sy):
    return np.sqrt(np.sum((dx - sx) ** 2 + (dy - sy) ** 2))


def _correspondence(a_kt, b_i, dx, dy, sx, sy, w1, w2):
    corr = (w1 * rmse(a_kt, b_i)) + (w2 * _fragment_distance(dx, dy, sx, sy))
    return corr


def _perform_fractal_exhaustive_search_2x1(range_partitions, transformed_domain_partitions_single, w1, w2):
    """ Perform the search for the best transform for each range fragment from the transformed domain image fragment """
    codebook = []
    for dx, dy, R in range_partitions:
        min_err = np.inf
        best_code = None
        for sx, sy, ti, D_T in transformed_domain_partitions_single:
            err = _correspondence(D_T, R, dx, dy, sx, sy, w1, w2)
            if err < min_err:
                min_err = err
                best_code = [dx, dy, sx, sy, ti, D_T]

        # Find colorimetric contraction
        dx, dy, sx, sy, ti, D_T = best_code
        C, B = _color_contraction(R, D_T)

        # 2x1 encodings are essentially the degenerate case of image composition that's always equivalent to
        # the domain image. Using our encoding scheme, this is Composition Method #0, i.e. ImageCompIdentityLeft
        # X = 0
        best_comp = find_best_image_composition(D_T, R, R, sim_metric='CW_SSIM')
        X = best_comp.index
        G = D_T.shape[0]    # Grid size. Assume images are square
        best_code = [dx, dy, sx, sy, ti, C, B, X, G]

        codebook.append(best_code)

    return codebook


def _perform_fractal_exhaustive_search_3x1(range_partitions, transformed_domain_partitions_pairs, w1, w2):
    """ Perform the search for the best transform for each range fragment from the transformed domain image fragment """
    codebook = []
    for dx, dy, R in range_partitions:

        search_results1 = []
        search_results2 = []
        Result = namedtuple('Result', ['err', 'code', 'img1', 'img2'])
        for sx, sy, ti, D1_T, D2_T in transformed_domain_partitions_pairs:
            err1 = _correspondence(D1_T, R, dx, dy, sx, sy, w1, w2)
            err2 = _correspondence(D2_T, R, dx, dy, sx, sy, w1, w2)
            search_results1.append(Result(err1, [dx, dy, sx, sy, ti], D1_T, D2_T))
            search_results2.append(Result(err2, [dx, dy, sx, sy, ti], D1_T, D2_T))

        best1 = min(search_results1, key=lambda x: x.err)
        best2 = min(search_results2, key=lambda x: x.err)
        best = min([best1, best2], key=lambda x: x.err)
        dx, dy, sx, sy, ti = best.code

        # Find best image composition
        best_comp = find_best_image_composition(best.img1, best.img2, R, sim_metric='CW_SSIM', show=False, width=2)
        D = best_comp.img_comp
        X = best_comp.index

        # Find colorimetric contraction
        C, B = _color_contraction(R, D)
        G = D.shape[0]  # Grid size. Assume images are square
        best_code = [dx, dy, sx, sy, ti, C, B, X, G]

        codebook.append(best_code)

    return codebook


def _set_num_process_workers(cpus):
    """ Sets the number of CPUs based on user inputs.
    If cpus == None, then all available cores are used (found by inspecting system resources using multiprocessing lib)
    """
    cpus = cpus if cpus is not None else get_cpu_count()
    assert isinstance(cpus, int) and cpus > 0, "Number of CPUs to use must be int > 0"
    # TODO: Will the autograder let me check to see how many worker processes I can use????
    if cpus > get_cpu_count():
        cpus = get_cpu_count()    # // 2  # Only use half of the cpus available
    return cpus


def fractal_encode(domain_images: Union[np.ndarray, Tuple, List], range_image: np.ndarray, grid_size: int, cpus=None, verbose=False):
    """ Perform the fractal encoding of the range image in terms of the given domain image(s). """

    # Setup params for multiprocessing
    cpus = _set_num_process_workers(cpus)
    print(f'***** Running fractal encoding using {cpus} threads *****')

    if isinstance(domain_images, np.ndarray):
        domain_images = [domain_images]

    if isinstance(domain_images, (tuple, list)) and len(domain_images) == 1:
        mode = '2x1'
        transformed_domain_partitions = _get_transformed_domain_partitions_single_image(domain_images[0], grid_size)
    elif isinstance(domain_images, (tuple, list)) and len(domain_images) == 2:
        mode = '3x1'
        transformed_domain_partitions = _get_transformed_domain_partitions_image_pair(domain_images[0], domain_images[1], grid_size)
    else:
        raise ValueError('Domain images arg must be a tuple or list containing a single image or two images!')

    range_partitions = _get_range_image_partitions(range_image, grid_size)

    # Hardcoded to split fragment search into # of jobs roughly equivalent to number of worker processes
    # This seems to work well based on cProfiling of the performance
    chunk_size = int(np.ceil(len(range_partitions) / cpus))

    # Problems like Basic Problem C-08 contain images that are all whitespace. We'll have to handle this by forcing
    # some sort of default encoding in this case
    num_range_blocks = range_image.shape[0] // grid_size
    if len(transformed_domain_partitions) > 0:

        w1, w2 = _calculate_correspondence_weights(transformed_domain_partitions, range_partitions)
        print('Using weights to calculate correspondence: (w1={}, w2={:.3f})'.format(w1, w2))

        if cpus == 1:
            range_partitions_chunks = [range_partitions]
        else:
            range_partitions_chunks = [chunk for chunk in divide_chunks(range_partitions, n=chunk_size)]

        print(f'Total # of Range Partitions Fragments to Search: {len(range_partitions)} (Split into {len(range_partitions_chunks)} chunks, chunk_size={chunk_size})')
        print(f'Total # of Domain Partition Chunks to Search: {len(transformed_domain_partitions)}')
        print(f'Total # of candidates for exhaustive search: {len(range_partitions) * len(transformed_domain_partitions)}')

        with poolcontext(processes=cpus) as pool:
            if mode == '3x1':
                codebook_jobs = pool.map(partial(_perform_fractal_exhaustive_search_3x1, transformed_domain_partitions_pairs=transformed_domain_partitions, w1=w1, w2=w2), range_partitions_chunks)
            else:
                codebook_jobs = pool.map(partial(_perform_fractal_exhaustive_search_2x1, transformed_domain_partitions_single=transformed_domain_partitions, w1=w1, w2=w2), range_partitions_chunks)

        """ Flatten jobs, then reshape into 3D list (range_grid_blocks x range_grid_blocks x code).
            Code itself is a 7-tuple like (dx, dy, sx, sy, ti, B)
            dx  ==  Range fragment x-coord
            dy  ==  Range fragment y-coord
            sx  ==  Domain fragment x-coord
            sy  ==  Domain fragment y-coord
            ti  ==  Best transform index number
            B   ==  Best brightness value
        """
        codebook = []
        for job in codebook_jobs:
            codebook += job

    else:
        codebook = _create_blank_image_codebook(range_partitions)

    codebook = [c for c in divide_chunks(codebook, num_range_blocks)]
    print(f'Final Codebook Total Length: {len(codebook) * len(codebook[0])}')
    return codebook


def numpy_fractal_encode_cache_map(*args, **kwargs):
    """ LRU cache implementation for functions whose FIRST parameter is a numpy array
    Provides big speedup for performing fractal encoding by caching results of computation

    Modified from source:
        https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
    """

    def decorator(function):
        @wraps(function)
        def wrapper(fractal_args_tuple: CachedFractalEncodeArgs):

            assert isinstance(fractal_args_tuple.domain_images, (tuple, list)), 'Cached fractal arg domain images must be tuple or list of image(s)'
            domain_images_hashable = tuple(map(tuple, [tuple(map(tuple, img)) for img in fractal_args_tuple.domain_images]))

            fractal_args_tuple_hashable = CachedFractalEncodeArgs(
                domain_images=domain_images_hashable,
                range_image=tuple(map(tuple, fractal_args_tuple.range_image)),
                grid_size=fractal_args_tuple.grid_size
            )
            return cached_wrapper(fractal_args_tuple_hashable, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(fractal_args_tuple_hashable: CachedFractalEncodeArgs, *args, **kwargs):

            domain_images_hashable = [np.array(img) for img in fractal_args_tuple_hashable.domain_images]
            frac_args_tuple = CachedFractalEncodeArgs(
                domain_images=domain_images_hashable,
                range_image=np.array(fractal_args_tuple_hashable.range_image),
                grid_size=fractal_args_tuple_hashable.grid_size
            )
            return function(frac_args_tuple, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator


CachedFractalEncodeArgs = namedtuple('CachedFractalEncodeArgs', ['domain_images', 'range_image', 'grid_size'])
CachedFractalDecodeArgs = namedtuple('CachedFractalDecodeArgs', ['codebook', 'source_images', 'target_image', 'num_iterations', 'show'])


@numpy_fractal_encode_cache_map()
def fractal_encode_cached(fractal_args_tuple: CachedFractalEncodeArgs):
    """ Perform the fractal encoding of the range image in terms of the given domain image. """
    codebook = fractal_encode(fractal_args_tuple.domain_images, fractal_args_tuple.range_image,
                              fractal_args_tuple.grid_size)
    return codebook


def _pad_image_to_shape(img1, reference_shape):

    h1, w1 = img1.shape
    h2, w2 = reference_shape

    dh = int(np.abs(h1 - h2))
    pt = int(np.abs(h1 - h2)) // 2
    pb = dh - pt

    dw = int(np.abs(w1 - w2))
    pl = int(np.abs(w1 - w2)) // 2
    pr = dw - pl

    if h1 - h2 < 0:
        img1 = np.pad(img1, ((pt, pb), (0, 0)), mode='constant', constant_values=255)
    else:
        img1 = img1

    if w1 - w2 < 0:
        img1 = np.pad(img1, ((0, 0), (pl, pr)), mode='constant', constant_values=255)
    else:
        img1 = img1

    return img1


def fractal_decode(codebook: list, source_images: Union[np.ndarray, Tuple, List], target_image: np.ndarray, num_iterations=5, show=False):
    """ Decode an image from an arbitrary source into the target image using a fractal codebook.

    Decoding isn't strictly speaking necessary for the project code, but it's really interesting to visualize (and
    super cool to watch the original image recovered from pure noise!)

    Defaults to using white noise image as source to decode from. Can pass in a source_image to decode an image
    using a different source (for instance, for an Raven's Progressive Matrix problem, decode image B using image A
    as the source, which would just be the inversion of the compression step when we encode image B in terms of image A.

    Modified from source:
        https://github.com/pvigier/fractal-image-compression/blob/master/compression.py
    """
    height, width = target_image.shape
    domain_grid_size = range_grid_size = height // len(codebook)

    if isinstance(source_images, np.ndarray):
        mode = '2x1'
        source_image1, source_image2 = [source_images, source_images]
    elif isinstance(source_images, (tuple, list)) and len(source_images) == 1:
        source_image1 = source_images[0]
        source_image2 = source_images[0]
    else:
        mode = '3x1'
        source_image1, source_image2 = source_images

    iterations1 = [resize_image(source_image1, (height, width))]
    iterations2 = [resize_image(source_image2, (height, width))]

    img_iter = np.zeros((height, width))

    """ In the event that the image isn't evenly divisible by the grid size, we end up with a remainder at the image
    borders that isn't covered by the decoding process. Since we're using a zero-valued canvas (i.e. black) to 
    reconstruct the image, we end up with a sharp discontinuity at the image edges because our RPM image problems have
    a white background and shapes are black, in opposition to the canvas. This has a big impact on our RMSE/SSIM 
    similarity calculation, so we'll need to fix this by forcing the remainder at the border to be padded white instead
    of black.

    Note that it's tempting to just invert the colors, but similarity(black_objects) != similarity(white_objects),
    so empirically this has proved to be a bad idea. This padding method is hacky, but seems to work much better.
    """
    w_remainder = width - (width // domain_grid_size) * domain_grid_size
    h_remainder = height - (height // domain_grid_size) * domain_grid_size
    img_iter[(height - h_remainder):, :] = 255
    img_iter[:, (width - w_remainder):] = 255

    for iter_num in range(num_iterations):
        for i in range(len(codebook)):
            for j in range(len(codebook[0])):
                dx, dy, sx, sy, ti, C, B, X, G = codebook[i][j]
                T = TRANSFORMS[ti]

                D1 = iterations1[-1][sx:sx + domain_grid_size, sy:sy + domain_grid_size]
                D2 = iterations2[-1][sx:sx + domain_grid_size, sy:sy + domain_grid_size]

                D = apply_composition(D1, D2, comp_name=X)

                D_t = C * T(D) + B
                img_iter[dx:dx + range_grid_size, dy:dy + range_grid_size] = D_t

        # Blurring seems to have mixed results, sometimes better, sometimes worse
        # from RavensImageOps import blur_image
        # img_iter = blur_image(img_iter, kernel_size=3, sigma=1.5)
        iterations1.append(img_iter)
        iterations2.append(img_iter)

        img_iter = np.zeros((height, width))
        # See previous note about padding remainder of image with white border instead of black
        w_remainder = width - (width // domain_grid_size) * domain_grid_size
        h_remainder = height - (height // domain_grid_size) * domain_grid_size
        img_iter[(height - h_remainder):, :] = 255
        img_iter[:, (width - w_remainder):] = 255

    def plot_iterations():
        try:
            import matplotlib.pyplot as plt
            from SSIM import cw_ssim

            # Configure plot
            fig = plt.figure()
            nb_row = int(np.ceil(np.sqrt(len(iterations1))))
            nb_cols = int(nb_row)
            # for i, img in enumerate(iterations1):  # Iterations should converge to same, so just use #1
            iterations_both = list(zip(iterations1, iterations2))
            for i, (img1, img2) in enumerate(iterations_both):
                ax = fig.add_subplot(nb_row, nb_cols, i + 1)

                if i == 0:
                    ax.imshow(iterations_both[0][0], cmap='gray', vmin=0, vmax=255, interpolation='none')
                    ax.set_title('Source #' + str(i))
                elif i == 1:
                    ax.imshow(iterations_both[0][1], cmap='gray', vmin=0, vmax=255, interpolation='none')
                    ax.set_title('Source #' + str(i))

                # Iterations should converge to same, so just use #1
                else:
                    # Display the error/similarity metric
                    # ax.set_title(str(i) + ' (' + 'RMSE={0:.3f}'.format(rmse(target_image, img) / img.size) + ')')
                    ax.imshow(img1, cmap='gray', vmin=0, vmax=255, interpolation='none')
                    ssim, _ = cw_ssim(target_image, img1)
                    ax.set_title(str(i) + ' (' + 'CW-SSIM={0:.3f}'.format(ssim) + ')')

                    frame = plt.gca()
                    frame.axes.get_xaxis().set_visible(False)
                    frame.axes.get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show()
            plt.close()
        except ImportError as e:
            import warnings
            warnings.warn('Matplotlib not available in current environment. Skipping image transform preview.')

    if show:
        plot_iterations()

    return iterations1


def get_mutual_fractals_features(problem: RavensProblem, rel: str, grid_size: int) -> set:

    r = rel.split(':')
    assert len(r) == 2 or len(r) == 3, "Can only accommodate pair or triplet relationships!"

    imgA = problem.figures[r[0]].img
    imgB = problem.figures[r[1]].img

    if len(r) == 2:
        T_AB = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgA], imgB, grid_size)))
        T_BA = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgB], imgA, grid_size)))
        feats = feature_union([T_AB, T_BA])
    else:
        imgC = problem.figures[r[2]].img
        T_AB = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgA, imgB], imgC, grid_size)))
        T_BA = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgB, imgA], imgC, grid_size)))
        T_BC = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgB, imgC], imgA, grid_size)))
        T_CB = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgC, imgB], imgA, grid_size)))
        T_AC = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgA, imgC], imgB, grid_size)))
        T_CA = features_from_fractals(fractal_encode_cached(CachedFractalEncodeArgs([imgC, imgA], imgB, grid_size)))

        featsAB = feature_union([T_AB, T_BA])
        featsBC = feature_union([T_BC, T_CB])
        featsAC = feature_union([T_AC, T_CA])
        feats = feature_union([featsAB, featsBC, featsAC])

    return feats


def get_problem_fractal_similarity(problem: RavensProblem, analogy: str, grid_size: int) -> float:
    """ Analogy should be like, for example A:B::C:1 or A:B:C::G:H:1 """
    prem, conc = analogy.split('::')
    feats_mat = get_mutual_fractals_features(problem, prem, grid_size)
    feats_alt = get_mutual_fractals_features(problem, conc, grid_size)
    # print('# Mat Feats', len(feats_mat))
    # print('# Alt Feats', len(feats_alt))
    S = tversky_feature_similarity(feats_mat, feats_alt)
    return S


def get_problem_fractal_similarity_multiscale(problem: RavensProblem, analogy: str, grid_sizes: Union[Tuple, List]) -> float:
    """ Analogy should be like, for example A:B::C:1 or A:B:C::G:H:1 """
    prem, conc = analogy.split('::')
    feats_mat = []
    feats_alt = []
    for gs in grid_sizes:
        feats_mat += get_mutual_fractals_features(problem, prem, gs)
        feats_alt += get_mutual_fractals_features(problem, conc, gs)
    # print('# Mat Feats', len(feats_mat))
    # print('# Alt Feats', len(feats_alt))
    S = tversky_feature_similarity(feats_mat, feats_alt)
    return S
