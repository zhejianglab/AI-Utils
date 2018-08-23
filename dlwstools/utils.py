import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import json
import random
import sys
from operator import itemgetter
import urllib.request
import shutil
import requests

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def process_ncrop_result( result, ncrop):
    nx, ny = result.shape
    nsize = nx // ncrop
    ret = np.zeros( (nsize, ny) ) 
    for i in range( nsize):
        ret[i,:] = np.mean( result[i*ncrop:(i+1)*ncrop, :], axis=0 )
        # if i == 0:
        #     print( "Result %d X %d Average %s to %s" % ( nx, ny, result[i*ncrop:(i+1)*ncrop, :], ret[i,:] ) )
    return ret

class ImageReader:
    def __init__(self, rootdir, dim_ordering='default', **kwargs ):
        self.rootdir = rootdir
        self.kwargs = kwargs
        self.dim_ordering = dim_ordering
    def read( self, file ):
        image = load_img( os.path.join(self.rootdir, file), **self.kwargs )
        img = img_to_array( image, dim_ordering = self.dim_ordering )
        # print (img.shape )
        return img

def read_in_images( rootdir, pattern=".*.jpg", pool = None, dim_ordering='default', **kwargs ):
    ex = re.compile( pattern )
    filelist = []
    for file in os.listdir(rootdir):
        if ex.match( file ):
            filelist.append(file)
    filelist.sort()
    if pool:
        imgReader = ImageReader( rootdir, dim_ordering, **kwargs ) 
        images = pool.map( imgReader.read, filelist )
    else:
        images = []
        for f in filelist:
            image = load_img( os.path.join(rootdir, f), **kwargs)
            img = img_to_array( image, dim_ordering = dim_ordering )
            # print (img.shape )
            images.append( img )
    return filelist, np.array( images )

def print_layers( model, first = None, last = None ):
    nlayers = len( model.layers )
    idx = 0
    for layer in model.layers:
        bPrint = True
        if first or last:
            bPrint = False
            if first and idx < first:
                bPrint = True
            if last and idx >= nlayers - last:
                bPrint = True
        if bPrint:
            print ( "Layer %d ==== %s" % (idx, layer.name ) )
        idx += 1

def download_if_notexist( file, file_uri ):
    if not os.path.exists(file):
        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(file_uri) as response, open(file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
def download_if_notexist( file, file_uri, token = None, cookies = None ):
    if not os.path.exists(file):
        # Download the file from `url` and save it locally under `file_name`:
        if token is None:
            if cookies is None:
                with urllib.request.urlopen(file_uri) as response, open(file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                    return
            else:
                response = requests.get( file_uri, cookies=cookies)
        else:
            response = requests.get( file_uri, headers={'Authorization': 'access_token %s' % token})
        if response.status_code == 200:
            data = response.raw.read()
            with open(file, 'wb') as f:
                f.write(data)
                print( "Write %s" % file ) 
        else:
            print(response.content)
        
class Tee(object):
    def __init__(self, name):
        print( "Tee started, output buffered to %s" %name )
        self.file = open(name, "w", 1)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        print( "Tee stopped" )
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


