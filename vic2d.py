import csv
import numpy as np
from PIL import Image
import re, os
import h5py
from collections import defaultdict

def read_csv_export_old(fpath, impath, keys=['exx', 'eyy', 'exy']):
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

def read_csv_export(fpath):
    """Read a csv file from vic2d into arrays.

    Inputs
    ------
    fpath : string
        Path to strain csv file exported from Vic-2D.

    Note
    ----
    It is assumed that Vic-2D uses 0-indexing for x and y.  The
    tooltip gives zero-indexed coordinates, and they match the csv
    export x and y coordinates, so this assumption seems valid.

    """
    data = defaultdict(list)
    with open(fpath, 'rb') as f:
        csvreader = csv.DictReader(f, skipinitialspace=True,
                                   delimiter=',')
        for row in csvreader:
            for k in csvreader.fieldnames:
                if row[k] is None:
                    msg = 'Missing values in line {} ' \
                          'in {}.'.format(csvreader.line_num,
                                          os.path.abspath(fpath))
                    raise Exception(msg)
                if len(row) > len(csvreader.fieldnames):
                    msg = 'Extra values in line {} ' \
                          'in {}.'.format(csvreader.line_num,
                                          os.path.abspath(fpath))
                    raise Exception(msg)
                try:
                    val = int(row[k])
                except ValueError:
                    val = float(row[k])
                data[k].append(val)
    return data

def hdf5ify(files, hdf5file):
    """Read a set of Vic-2D csv files and save them all as hdf5

    """
    with h5py.File(hdf5file, 'w') as h5f:
        for csvfile in files:
            fstr = os.path.basename(csvfile)
            nm, ext = os.path.splitext(fstr)
            dgroup = h5f.create_group(nm)
            data = read_csv_export(csvfile)
            for k in data:
                dgroup.create_dataset(k, data=data[k])
