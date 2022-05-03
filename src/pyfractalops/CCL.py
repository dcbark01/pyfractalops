from collections import namedtuple

import numpy as np

from pyfractalops.utils import sobel_edge_detector, intersection2d


# Simple class for creating our object graph tree structure
Vertex = namedtuple('Vertex', ['value', 'parent', 'rank'])


class Node(object):
    """ Mofified from source: https://github.com/jacklj/ccl/blob/gh-pages/node.py"""

    def __init__(self, x):
        self.parent = self
        self.value = x
        self.size = 0

    # Print format for debugging
    def __str__(self):
        st = "[value: " + str(self.value) + ", parent: " + str(self.parent.value)
        st += ", size: " + str(self.size) + "]"
        return st


class DisjointSet(object):
    """ Find objects/shapes in an input image array using the connected component labeling algorithm.
        Implemented using the following references:
            1. https://en.wikipedia.org/wiki/Connected-component_labeling
            2. https://github.com/jacklj/ccl/blob/gh-pages/ccl.py
            3. https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """

    def __init__(self, elements=None):
        self.elements = elements if elements is not None else {}

    def make_set(self, x):
        if x in self.elements:
            return self.elements[x]
        else:
            node = Node(x)
            self.elements[x] = node
        return node

    def find(self, v):
        if v.parent != v:
            v.parent = self.find(v.parent)
            return v.parent
        else:
            return v

    def union(self, v1, v2):
        if v1 == v2:
            return

        v1_root = self.find(v1)
        v2_root = self.find(v2)
        if v1_root == v2_root:
            return

        if v1_root.size > v2_root.size:
            v2_root.parent = v1_root
        elif v1_root.size < v2_root.size:
            v1_root.parent = v2_root
        else:
            v1_root.parent = v2_root
            v2_root.size += 1

    def get_node(self, val):
        pass

    def display_all_nodes(self):
        for e in self.elements.values():
            print(e)

    def display_all_sets(self):
        sets = {}  # Initialise so nodes can't be added twice

        # Add all nodes to set dictionary
        # keys    :=   representative element of each set
        # values  :=   the elements of the set with that representative
        for item in self.elements.values():
            if self.find(item).value not in sets.keys():
                sets[self.find(item).value] = []  # initialise list for this key
            sets[self.find(item).value].append(item)

        # Display each representative key's set of items
        st = ""
        for representative in sets.keys():
            st = st + "("
            for item in sets[representative]:
                st = st + str(item.value) + ","
            st = st[:-1]  # remove final ','
            st = st + ") "
        print(st)


def label_connected_components(img_arr):
    """ Find objects/shapes in an input image array using the connected component labeling algorithm.
    Implemented using the following references:
        1. https://en.wikipedia.org/wiki/Connected-component_labeling
        2. https://github.com/jacklj/ccl/blob/gh-pages/ccl.py
        3. https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    NOTE: This algorithm expects a binary input image where objects of interest are WHITE (ie 1) and background is
    black (ie 0)
    """

    # Add padding to image to account for neighborhood calculation at boundaries
    img = np.pad(img_arr, (1, 1), mode='edge')
    h, w = img.shape[0:2]
    linked = []
    img_labelled = np.zeros_like(img)
    current_label = 1    # Label counter

    # First pass
    dsj = DisjointSet()
    for j in range(h - 2):
        for i in range(w - 2):

            x = i + 1
            y = j + 1
            p = img[y, x]

            if p == 0:
                # Must be background pixel - ignore
                pass
            else:
                # Must be foreground pixel
                labels = _get_neighbors(img_labelled, x, y)
                # print(labels)

                if not labels:
                    img_labelled[y, x] = current_label
                    dsj.make_set(current_label)

                    current_label += 1
                else:
                    label_min = min(labels)
                    img_labelled[y, x] = label_min

                    if len(labels) > 1:  # More than one type of label found --> add equivalence class
                        for lbl in labels:
                            dsj.union(dsj.elements[label_min], dsj.elements[lbl])

    # Second pass: Replace labels with root labels
    labels_final = {}
    new_label_num = 1

    h, w = img_labelled.shape[0:2]
    for y in range(h):
        for x in range(w):

            p = img_labelled[y, x]
            if p > 0:   # Foreground pixel
                # Get element set's representative value and use as pixel's new label
                new_label = dsj.find(dsj.elements[p]).value
                img_labelled[y, x] = new_label

                # Add label to list of labels used, for 3rd pass (flattening label list)
                if new_label not in labels_final:
                    labels_final[new_label] = new_label_num
                    new_label_num += 1

    # Third pass: final loop to make object labels match root labels
    h, w = img_labelled.shape[0:2]
    for y in range(h):
        for x in range(w):
            p = img_labelled[y, x]

            if p > 0:  # Foreground pixel
                img_labelled[y, x] = labels_final[p]

    # Get rid of extra padding that we added when calculating neighborhood
    img_labelled = img_labelled[1:-1, 1:-1]
    return img_labelled


def _get_neighbors(img_arr, x, y):

    labels = set()

    w = img_arr[y, x-1]
    if w > 0:
        labels.add(w)
    n = img_arr[y-1, x]
    if n > 0:
        labels.add(n)
    nw = img_arr[y-1, x-1]
    if nw > 0:
        labels.add(nw)
    ne = img_arr[y-1, x+1]
    if ne > 0:
        labels.add(ne)

    return labels


def fill(img):
    """ Fill the image/object/shape/roi in an image. """
    white = np.ones_like(img) * 255
    black = np.zeros_like(img)
    img_bin = np.where(img < 128, 1, 0).astype(np.int8)
    img_filled_bin = np.maximum.accumulate(img_bin, 1) & \
                     np.maximum.accumulate(img_bin[:, ::-1], 1)[:, ::-1] & \
                     np.maximum.accumulate(img_bin[::-1, :], 0)[::-1, :] &\
                     np.maximum.accumulate(img_bin, 0)
    img_filled = np.where(img_filled_bin, black, white)
    return img_filled


def unfill(img):
    """ Unfill the image/object/shape/roi in an image. """
    img_edges = sobel_edge_detector(img)
    img_unfilled = np.where(intersection2d(img, img_edges) > 128, 255, 0)
    return img_unfilled
