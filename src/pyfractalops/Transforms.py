import sys
import itertools
from collections import namedtuple
from itertools import chain, combinations

import numpy as np

from pyfractalops.CCL import unfill, fill


TRANSFORMS = [
    'Identity',
    'Rotate90',
    'Rotate180',
    'Rotate270',
    'HorizontalFlip',
    'VerticalFlip',
    'ReflectYnX',
    'ReflectYX',
    'ShapeFiller',
    'ShapeUnfiller'
]


class Transformation(object):

    def __init__(self, **kwargs):
        self.name = ''
        self.complexity = None
        self.attributes = {k: v for (k, v) in kwargs.items()}

    def __eq__(self, other):
        if not isinstance(other, Transformation):
            return False
        else:
            if self.name == other.name:
                return True
            else:
                return False

    def apply(self, img):
        """ Abstract method for applying the transform to the image (must be overriden by child classes).
        Should return a 2d numpy array grayscale image, e.g.

        img_t = <transform_function>(img)
        return img_t
        """
        raise NotImplementedError

    def preview_transform(self, img):
        img_t = self.apply(img)
        try:
            from Plotting import show_image_pair
            show_image_pair(img, img_t, suptitle=self.name)
        except ImportError as e:
            print('Matplotlib plotting not available in current env - skipping!')


# <editor-fold desc="********** BASIC AFFINE TRANSFORMATIONS **********">
class Identity(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 0

    def apply(self, img):
        img_t = img
        return img_t


class Rotate90(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 1
        self.scalar_val = 90

    def apply(self, img):
        img_t = np.rot90(img)
        return img_t


class Rotate180(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 1
        self.scalar_val = 180

    def apply(self, img):
        img_t = np.rot90(np.rot90(img))
        return img_t


class Rotate270(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 1
        self.scalar_val = 270

    def apply(self, img):
        img_t = np.rot90(np.rot90(np.rot90(img)))
        return img_t


class HorizontalFlip(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 2

    def apply(self, img):
        img_t = np.fliplr(img)
        return img_t


class VerticalFlip(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 2

    def apply(self, img):
        img_t = np.flipud(img)
        return img_t


class ReflectYnX(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 3

    def apply(self, img):
        img_t = np.rot90(np.flipud(img))
        return img_t


class ReflectYX(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 3

    def apply(self, img):
        img_t = np.rot90(np.fliplr(img))
        return img_t
# </editor-fold>


# <editor-fold desc="********** TRANSLATION OPERATIONS **********">
def translate2d(img_arr, tx, ty, constant=0):
    """ Shifts 2d grayscale image array (H x W) by (tx, ty)
    Values along edge are padded with value specified by 'constant' arg
    """

    h, w = img_arr.shape[0:2]

    # Limit translation to max height width of shape
    if tx > w:
        tx = w
    elif abs(tx) > w:
        tx = -w

    if ty > h:
        ty = h
    elif abs(ty) > h:
        ty = -h

    # Translate on X-axis first
    img_tr = img_arr
    img_tr = np.roll(img_tr, tx, axis=1)
    if tx < 0:
        img_tr[:, tx:] = constant
    elif tx > 0:
        img_tr[:, 0:tx] = constant
    else:
        img_tr = img_tr

    # Now translate Y-axis
    img_tr = np.roll(img_tr, ty, axis=0)
    if ty < 0:
        img_tr[ty:, :] = constant
    elif ty > 0:
        img_tr[0:ty, :] = constant
    else:
        img_tr = img_tr
    return img_tr


class Translation(Transformation):

    def __init__(self, tx, ty, constant=1):
        """ Shift an image by (tx, ty) pixels.
        Note that the 'constant' arg specifies how the image border will padded. If the RPM problem consists of black
        objects (i.e. pixels with value 0 are objects), then the constant arg should be 1 to ensure the border is padded
        with a white background. Conversely, constant should be == 0 if image objects are hi pixel vals (i.e. 1 or 255,
        typical if the colors have been inverted so that objects are 'True' pixel vals rather than False vals) so that
        the border is padded with a black background.

        """
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 3  # TODO: Not sure what the right complexity value should be for this

        self.tx = tx
        self.ty = ty
        self.constant = constant

        # Attributes just used for pretty string printing when composing translations
        self.attributes = {'tx': self.tx, 'ty': self.ty}

    def apply(self, img):
        img_t = translate2d(img, self.tx, self.ty, constant=self.constant)
        return img_t

# </editor-fold>


# <editor-fold desc="********** SHAPE FILL/UNFILL OPERATIONS **********">
class ShapeFiller(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 5  # TODO: Not sure what the right complexity value should be for this

    def apply(self, img: np.ndarray):
        img_t = fill(img)
        return img_t


class ShapeUnfiller(Transformation):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.complexity = 5  # TODO: Not sure what the right complexity value should be for this

    def apply(self, img: np.ndarray):
        img_t = unfill(img)
        return img_t

# </editor-fold>


class Compose(object):
    """ Class for composing chains of image transformations.

    Modified from original source here:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Compose

    You can compose a set of transforms like so:

        idnty = Identity()
        print(idnty.name)
        idnty.preview_transform(img1)

        rot90 = Rotate90()
        print(rot90.name)
        rot90.preview_transform(img1)

        rot180 = Rotate180()
        print(rot180.name)
        rot180.preview_transform(img1)

        rot270 = Rotate270()
        print(rot270.name)
        rot270.preview_transform(img1)

        hflip = HorizontalFlip()
        print(hflip.name)
        hflip.preview_transform(img1)

        vflip = VerticalFlip()
        print(vflip.name)
        vflip.preview_transform(img1)


        transform = Compose([
            VerticalFlip(),
            Rotate90(),
            HorizontalFlip(),
            Identity()
        ])


        img2 = transform(img)
        print('Testing Transform Composer')
        print(transform)
        preview_transform(img1, img2)
    """
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms is not None else []

    def __getitem__(self, item):
        return self.transforms[item]

    def __eq__(self, other):

        check_idnty = lambda x: True if x == 'Identity' else False
        T_idnty = [check_idnty(t.name) for t in self.transforms]
        T_idnty_other = [check_idnty(to.name) for to in other.transforms]

        # If all are identity, no transformation is made so compositions are equivalent
        if all(T_idnty) and all(T_idnty_other):
            return True
        else:

            # Once identity transforms are eliminated, if two compositions aren't of same length can't be equivalent
            t1 = [t for t in self.transforms if t.name != 'Identity']
            t2 = [t for t in other.transforms if t.name != 'Identity']
            if len(t1) != len(t2):
                return False
            else:
                # Check transform by transform for equality,
                is_match = []
                for f1, f2 in list(zip(t1, t2)):
                    if f1 == f2:
                        is_match.append(True)
                    else:
                        is_match.append(False)

                # If every transform matches, exclusive of identity, then composition is equivalent
                if all(is_match):
                    return True
                else:
                    return False

    def __call__(self, img):
        for t in self.transforms:
            img = t.apply(img)
        return img

    def __len__(self):
        return len(self.transforms)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '{0}('.format(t.name)
            format_string += ', '.join([f'{k}={v}' for (k, v) in t.attributes.items()])
            format_string += ')'
        format_string += ')'
        return format_string

    def decode(self):
        """ Return a tuple representation of the transforms in the composition. """
        avail_transforms = [t for t in TRANSFORMS]

        code_vals = {k: 0 for k in avail_transforms}
        CodeTuple = namedtuple('CodeTuple', code_vals)

        for t in self.transforms:
            if t.name in avail_transforms:
                # Rotations will have scalar vals, else just set flag to binary 0/1
                val = getattr(t, 'scalar_val', 1)
            else:
                val = 0
            code_vals[t.name] = val
        code = CodeTuple(**code_vals)
        return code

    @property
    def complexity(self):
        """ Return the overall complexity score for the given combination of transforms. """
        return sum([t.complexity for t in self.transforms])

    def append(self, transform):
        self.transforms.append(transform)


def apply_transform(img, A, HF, VF, RYX, RYnX, FILL):
    """ Apply the specified transform to the input image.

    :param img:     ndarray (H x W) grayscale image
    :param A:       angle
    :param HF:      horizontal flip
    :param VF:      vertical flip
    :param RYX:     reflect yx
    :param RYnX:    reflect ynx
    :param FILL:    fill/unfill
    :return: img_t  ndarray (H x W) transformed grayscale image
    """
    if A == 0:
        T_angle = Identity()
    elif A == 90:
        T_angle = Rotate90()
    elif A == 180:
        T_angle = Rotate180()
    elif A == 270:
        T_angle = Rotate270()
    else:
        raise ValueError("Only valid rotation angles are 0, 90, 180, 270")

    if HF == 0:
        T_HF = Identity()
    elif HF == 1:
        T_HF = HorizontalFlip()
    else:
        raise ValueError("Only valid horizontal flips are 0 (Identity) or 1. ")

    if VF == 0:
        T_VF = Identity()
    elif VF == 1:
        T_VF = VerticalFlip()
    else:
        raise ValueError("Only valid vertical flips are 0 (Identity) or 1. ")

    if RYX == 0:
        T_RYX = Identity()
    elif RYX == 1:
        T_RYX = ReflectYX()
    else:
        raise ValueError("Only valid YX reflections are 0 (Identity) or 1. ")

    if RYnX == 0:
        T_RYnX = Identity()
    elif RYnX == 1:
        T_RYnX = ReflectYnX()
    else:
        raise ValueError("Only valid YnX reflections are 0 (Identity) or 1. ")

    if FILL == 0:
        T_FILL = Identity()
    elif FILL == 1:
        T_FILL = ShapeUnfiller()
    elif FILL == 2:
        T_FILL = ShapeFiller()
    else:
        raise ValueError("Only valid fill/unfill operations are 0 (Identity), 1 (Unfill), or 2 (Fill). ")

    T = Compose([])
    for transform in [T_angle, T_HF, T_VF, T_RYX, T_RYnX, T_FILL]:
        T.append(transform)

    img_t = T(img)
    return img_t


def str2class(name):
    """ Get a class instance by calling by string name.
    Source: https://stackoverflow.com/questions/17959996/get-python-class-object-from-class-name-string-in-the-same-module
    """
    return getattr(sys.modules[__name__], name)


def powerset(iterable):
    """ Generate sets of possible combinations from iterable.
    Example: "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    Modified from source found on StackOverflow:
    https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
    """
    s = list(iterable)  # allows duplicate elements
    combos = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return combos


def create_candidate_transforms(max_n=3, max_complexity=10):
    """ Generate a list of candidate transform combinations that meet the input criteria. """
    combos = [c for c in powerset(TRANSFORMS) if c != () and len(c) <= max_n]
    candidates = []
    for c in combos:
        t = []
        for tname in c:
            t.append(str2class(tname)())
        transform = Compose(t)
        if transform.complexity < max_complexity:
            candidates.append(transform)
    return candidates


def create_candidate_translations(tx_vals, ty_vals):
    coords = [x for x in itertools.product(tx_vals, ty_vals)]
    translations = []
    for tx, ty in coords:
        Tr = Compose([
            Translation(tx=tx, ty=ty)
        ])

        translations.append(Tr)
    return translations
