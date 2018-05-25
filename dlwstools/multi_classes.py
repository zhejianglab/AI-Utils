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
            self.numclasses = self.truth.shape[1]
    
    def histogram(self):
        hist = np.zeros(self.numclasses)

        for (i,item) in enumerate(self.truth):
            #if i==0:
            #    print ("Item[0] is %s" % item )
            for (j, val) in enumerate(item):
                if item[j] > 0.5:
                    hist[j] += 1
        return hist

    # return recall, precision and f1 score.
    def score(self):
        _tt = np.zeros( self. num_classes )
        _tf = np.zeros( self. num_classes )
        _ft = np.zeros( self. num_classes )
        nitems = self.truth.shape[0]        
        for i in range(nitems):
            pred_class = {}
            org_class = {}
            for j in range(self.num_classes):
                if self.pred[i][j]>0.5:
                    pred_class[j] = True
                if self.truth.classes[i][j]>0.5:
                    org_class[j] = True
            pred_extra_class = {}
            for (j, _) in pred_class.items():
                if j in org_class:
                    _tt[j] += 1
                    org_class.pop(j)
                else:
                    pred_extra_class[j] = True
            for (j, _) in org_class.items():
                _tf[j] += 1
            for (j, _) in pred_extra_class.items():
                _ft[j] += 1
        recall = (_tt ) / ( _tt + _tf)
        recall = 1-np.nan_to_num( 1-recall )
        precision = (_tt) / (_tt + _ft )
        precision = 1-np.nan_to_num( 1-precision )
        f1 = 2 / ( (1/recall) + (1 / precision))

        return ( recall, precision, f1)

