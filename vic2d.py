import csv
import numpy as np
from PIL import Image
import re, os
from os import path
import h5py
from collections import defaultdict

def listcsvdir(directory):
    """List csv files in a directory.

    """
    files = sorted(os.listdir(directory))
    csvonly = (f for f in files if
               not f.startswith('.')
               and f.endswith('.csv'))
    abspaths = (path.abspath(path.join(directory, f)) for f in csvonly)
    return list(abspaths)
