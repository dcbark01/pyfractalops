import sys
from typing import Union
from collections import namedtuple

from pyfractalops.SSIM import CachedCWSSIMArgs, cw_ssim_cached, calculate_ssim
from pyfractalops.utils import union2d, intersection2d, subtraction2d, xor2d, rmse, tversky_image_similarity


COMPOSITION_TRANSFORMS = {
    'ImageCompIdentityLeft': 0,
    # 'ImageCompIdentityRight': 1,
    'ImageUnion': 2,
    'ImageIntersection': 3,
    'ImageSubtract': 4,
    'ImageBacksubtract': 5,
    'ImageXOR': 6
}
COMPOSITION_TRANSFORMS_INV = {v: k for (k, v) in COMPOSITION_TRANSFORMS.items()}


def str2class(name):
    """ Get a class instance by calling by string name.
    Source: https://stackoverflow.com/questions/17959996/get-python-class-object-from-class-name-string-in-the-same-module
    """
    return getattr(sys.modules[__name__], name)


def image_composition_selector(comp_name: Union[str, int]):
    if isinstance(comp_name, str):
        comp_name = comp_name
    else:
        comp_name = COMPOSITION_TRANSFORMS_INV[comp_name]
    return str2class(comp_name)()


class ImageComposition(object):

    def __init__(self, **kwargs):
        self.name = ''
        self.complexity = None
        self.attributes = {k: v for (k, v) in kwargs.items()}

    def __eq__(self, other):
        if not isinstance(other, ImageComposition):
            return False
        else:
            if self.name == other.name:
                return True
            else:
                return False

    def apply(self, img1, img2):
        """ Abstract method for applying the transform to the image (must be overriden by child classes).
        Should return a 2d numpy array grayscale image, e.g.

        img_t = <transform_function>(img)
        return img_t
        """
        raise NotImplementedError

    def preview_transform(self, img1, img2):
        img_t = self.apply(img1, img2)
        try:
            from Plotting import show_image_triplet
            show_image_triplet(img1, img2, img_t, suptitle=self.name)
        except ImportError as e:
            print('Matplotlib plotting not available in current env - skipping!')


class ImageUnion(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = union2d(img1, img2)
        return img_t


class ImageIntersection(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = intersection2d(img1, img2)
        return img_t


class ImageSubtract(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = subtraction2d(img1, img2)
        return img_t


class ImageBacksubtract(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = subtraction2d(img2, img1)
        return img_t


class ImageXOR(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = xor2d(img1, img2)
        return img_t


class ImageCompIdentityLeft(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = img1
        return img_t


class ImageCompIdentityRight(ImageComposition):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img1, img2):
        img_t = img2
        return img_t


def apply_composition(img1, img2, comp_name: Union[str, int]):
    """ Apply the specified transform to the input image.

    :param img:     ndarray (H x W) grayscale image
    :param A:       (str or int)
    """
    comp_fcn = image_composition_selector(comp_name)
    img_t = comp_fcn.apply(img1, img2)
    return img_t


def find_best_image_composition(img_src1, img_src2, img_tgt, sim_metric='CW_SSIM', width=5):

    result = namedtuple('Result', ['index', 'comp_name', 'score', 'img_comp'])
    results = []
    for comp_idx in COMPOSITION_TRANSFORMS_INV.keys():
        img_comp = apply_composition(img_src1, img_src2, comp_idx)

        if sim_metric.lower() == 'cw_ssim':
            mode = 'max'
            args = CachedCWSSIMArgs(img_comp, img_tgt, width=width)
            S, _ = cw_ssim_cached(args)
        elif sim_metric.lower() == 'ssim':
            mode = 'max'
            S, _ = calculate_ssim(img_comp, img_tgt)
        elif sim_metric.lower() == 'rmse':
            mode = 'min'
            S = rmse(img_comp, img_tgt) / img_comp.size
        elif sim_metric.lower() == 'tversky':
            mode = 'max'
            S = tversky_image_similarity(img_comp, img_tgt)
        else:
            raise ValueError('Only CW_SSIM (default), SSIM, RMSE, and Tversky Similarity metrics are supported!')

        results.append(result(index=comp_idx, comp_name=COMPOSITION_TRANSFORMS_INV[comp_idx], score=S, img_comp=img_comp))

    if mode == 'max':
        best_comp = max(results, key=lambda x: x.score)
    elif mode == 'min':
        best_comp = min(results, key=lambda x: x.score)
    else:
        raise ValueError('Only min and max metrics are supported!')

    return best_comp
