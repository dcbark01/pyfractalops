import numpy as np
from typing import Tuple, List, Union
from itertools import chain, combinations

from pyfractalops.Transforms import create_candidate_transforms


TRANSFORMS = create_candidate_transforms(max_n=1, max_complexity=10)


class FeatureVar(object):

    def __init__(self, var_name: str, val=None):
        """ Class for representing fractal features. """
        self.var_name = var_name
        self.val = val

    def __sub__(self, other):
        return FeatureVar(self.var_name, val=self.val - other.val)

    def __abs__(self):
        return FeatureVar(self.var_name, val=abs(self.val))

    def __add__(self, other):
        return FeatureVar(self.var_name, val=self.val + other.val)

    def __mul__(self, other):
        return FeatureVar(self.var_name, val=self.val * other.val)

    def __str__(self):
        """ Tag the feature using a string encoding of the various feature values. """
        enc_str = f'{self.var_name}={self.val}'
        return enc_str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if self.var_name == other.var_name and self.val == other.val:
            return True
        else:
            return False


class FeatureVarBrightness(FeatureVar):

    step = 20
    BRIGHTNESS_RANGES = np.arange(-260, 260, step)

    def __init__(self, var_name, val=None):
        super().__init__(var_name, val)

    def __str__(self):

        FeatureVarBrightness.BRIGHTNESS_RANGES[0] = -255
        FeatureVarBrightness.BRIGHTNESS_RANGES[-1] = 255
        idx = np.searchsorted(FeatureVarBrightness.BRIGHTNESS_RANGES, self.val)
        if self.val >= 255:
            idx = FeatureVarBrightness.BRIGHTNESS_RANGES.shape[0] - 1

        if self.val <= -255:
            idx = 1
        val_range = (FeatureVarBrightness.BRIGHTNESS_RANGES[idx - 1], FeatureVarBrightness.BRIGHTNESS_RANGES[idx])
        enc_str = f'{self.var_name}={val_range}'
        return enc_str


class Feature(object):

    def __init__(self, feat_name: str, feat_vars: Tuple):
        self.feat_name = feat_name
        self.feat_vars = feat_vars

    def __str__(self):
        return f'{self.feat_name}_' + '_'.join([str(fv) for fv in self.feat_vars])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        # if str(self) == str(other):
        if hash(self) == hash(other):
            return True
        else:
            return False


def powerset(iterable):
    """ Generate sets of possible combinations from iterable.
    Example: "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    Modified from source found on StackOverflow:
    https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
    """
    s = list(iterable)  # allows duplicate elements
    combos = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return combos


def features_from_fractals(codebook):
    """ Create feature subsets from a fractal codebook (see McGreggor 2012, 2014, Kunda, etc). """
    features = []
    for i in range(len(codebook)):
        for j in range(len(codebook[0])):
            dx, dy, sx, sy, ti, _, B, X, G = codebook[i][j]

            # Note: These values can cause pretty drastic differences if included in code book
            # See the original fractal papers for more info on these (not a whole lot of info to be found though)
            # B = 255
            # C = 0.75

            # Unpack the transformation composition
            T = TRANSFORMS[ti]
            id_val, r90_val, r180_val, r270_val, hf_val, vf_val, ryx_val, rynx_val, fill_val, unfill_val = T.decode()

            dx = FeatureVar('dx', val=dx)
            dy = FeatureVar('dy', val=dy)
            sx = FeatureVar('sx', val=sx)
            sy = FeatureVar('sy', val=sy)
            ident = FeatureVar('id', val=id_val)
            angle = FeatureVar('angle', val=max([r90_val, r180_val, r270_val]))
            hf = FeatureVar('hf', val=hf_val)
            vf = FeatureVar('vf', val=vf_val)
            ryx = FeatureVar('ryx', val=ryx_val)
            rynx = FeatureVar('rynx', val=rynx_val)
            fill = FeatureVar('fill', val=fill_val)
            unfill = FeatureVar('unfill', val=unfill_val)
            # C = FeatureVar('contrast', val=round(C, 2))    # Currently unused
            B = FeatureVarBrightness('brightness', val=B)
            # B = FeatureVar('brightness', val=B)
            X = FeatureVar('img_comp', val=X)
            G = FeatureVar('grid_size', val=G)

            feat_spec = Feature('feat_spec', (dx, dy, sx, sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, B, X, G))
            feat_pos_ag = Feature('pos_ag', (dx - sx, dy - sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, B, X, G))
            feat_grid_ag = Feature('grid_ag', (dx, dy, sx, sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, B, X))
            feat_comp_ag = Feature('comp_ag', (dx, dy, sx, sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, B, G))
            feat_comp_grid_ag = Feature('comp_grid_ag', (dx, dy, sx, sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, B))
            feat_bright_ag = Feature('bright_ag', (dx, dy, sx, sy, ident, angle, hf, vf, ryx, rynx, fill, unfill, X, G))
            feat_trans_ag = Feature('trans_ag', (dx, dy, sx, sy, B, X, G))

            feat_trans_spec = Feature('trans_spec', (ident, angle, hf, vf, ryx, rynx, fill, unfill))
            feat_bright_spec = Feature('bright_spec', (B,))
            feat_comp_spec = Feature('comp_spec', (X,))
            # feat_grid_spec = Feature('grid_spec', (G,))
            feat_comp_grid_spec = Feature('comp_grid_spec', (G, X))

            feat_pos_abs = Feature('feat_pos_abs', (abs(dx - sx), abs(dy - sy), ident, angle, hf, vf, ryx, rynx, fill, unfill, B, X, G))
            feat_pos_shift = Feature('feat_pos_shift', (dx - sx, dy - sy))

            features.append([
                feat_spec,
                feat_pos_ag,
                feat_grid_ag,
                feat_comp_ag,
                feat_bright_ag,
                feat_trans_ag,
                feat_trans_spec,
                feat_bright_spec,
                # feat_grid_spec,
                feat_comp_spec,
                feat_pos_abs,
                feat_pos_shift
            ])

    return [str(feature_set) for ft in features for feature_set in ft]  # Flatten features int single list


def feature_union(feats: Union[Tuple, List]):
    return set().union(*feats)


def feature_intersection(feats: Union[Tuple, List]):
    return set(feats[0]).intersection(*feats)


def feature_subtraction(feats1, feats2):
    return set(feats1) - set(feats2)


def tversky_feature_similarity(feats, feats_other):
    """ Calculate feature similarity (hardcoded to alpha=beta=1.0, equivalent to Jaccard similarity). """
    s = len(feature_intersection([feats, feats_other])) / len(feature_union([feats, feats_other]))
    return s
