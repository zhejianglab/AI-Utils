"""
Common utility functions for multi classes.

Licensed under the MIT License (see LICENSE for details)
Written by Jin Li
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings


# Convert a sparse 2D array to a dense 2D array with a class map
def sparse_to_dense( origin, classmap):
    result = np.zeros( (origin.shape[0], len(classmap) )
    for i in range(origin.shape[0]):
        for j in origin[i]:
            result[i][classmap[j]] = 1
    return result


############################################################
#  MultiClass
############################################################
class MultiClass(object):
    """Instantiates a multiclass objects, used for data analysis of multiclass and
    result analysis.
    """

    def __init__(self, truth, pred=None, classmap=None):
        ()

