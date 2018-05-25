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
    nitems = len(origin)
    nclasses = len(classmap)
    result = np.zeros( (nitems, nclasses ) )
    for i in range(nitems):
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
        self.numclasses = None
        if classmap is not None:
            truth = sparse_to_dense( truth, classmap)
            self.numclasses = len(classmap)
            if pred is not None:
                pred = sparse_to_dense( pred, classmap)
        self.truth = truth
        self.pred = pred
        if self.numclasses is None:
            self.numclasses = self.truth.shape[0]
    
    def histogram(self):
        hist = np.zeros(self.numclasses)

        for (i,item) in enumerate(self.truth):
            #if i==0:
            #    print ("Item[0] is %s" % item )
            for (j, val) in enumerate(item):
                if item[j] > 0.5:
                    hist[j] += 1
        return hist
        


