import csv
import numpy as np
from PIL import Image
import re

def read_vic2d_export(fpath, impath, keys=['exx', 'eyy', 'exy']):
    """Read a csv file from vic2d into arrays.

    Inputs
    ------
    fpath : string
        Path to strain csv file exported from Vic-2D.
    impath : string
        Path to image corresponding to the strain file.

    Note
    ----
    It is assumed that Vic-2D uses 0-indexing for x and y.  The
    tooltip gives zero-indexed coordinates, and they match the csv
    export x and y coordinates.

    """
    img = Image.open(impath)
    sz = img.size
    data = {}
    for k in keys:
        data[k] = np.zeros(sz)
    with open(fpath, 'rb') as f:
        csvreader = csv.DictReader(f, skipinitialspace=True)
        for row in csvreader:
            i = int(row['x'])
            j = int(row['y'])
            for k in keys:
                try:
                    val = int(row[k])
                except ValueError:
                    val = float(row[k])
                data[k][i][j] = val
    return data
