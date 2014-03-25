# -*- coding: utf-8 -*-
"""
`choose_images` is a command line tool to sequester images from a
mechanical test that aren't needed for Vic2D analysis.  It uses
information from a correctly filled out `image_index.csv` choose which
images are useful.

Usage:

$ choose_images IMAGE_INDEX.CSV

"""

import os, re, sys, csv
import re
import numpy as np
import mechana as mech
from mechana import instron

def image_strain(imdir, mechcsv):
    """Returns a list of (image name, strain) tuples.

    The image names omit the file extension.

    The strain is based on Instron extension; not optical strain.

    """
    # Load image data
    imnames = []
    for fname in os.listdir(imdir):
        if fname.endswith((".tiff", ".tif")):
            imnames.append(fname)
    imnames.sort()
    imdict = read_image_index(os.path.join(imdir, 'image_index.csv'))
    reftime = image_time(imdict['reference'])
    imtimes = [image_time(n) - reftime for n in imnames]
    scale = image_scale(os.path.join(imdir, 'image_scale.csv'))

    # Calculate reference length
    l0 = reference_length(os.path.join(imdir, 'ref_length.csv'), scale)

    # Load mechanical test data
    t, d, p = mech.instron.read_instron_csv(mechcsv)
    with open(os.path.join(imdir, 'ref_time.csv')) as f:
        reader = csv.reader(f)
        mech_reftime = float(reader.next()[0])
    t = t - mech_reftime

    # Calculate stretch
    y = (d + l0) / l0

    # Interpolate stretch at image times
    y_im = np.interp(imtimes, t, y)

    return zip(imnames, y_im)

def image_scale(fpath):
    """Reads `image_scale.csv` and calculates mm/px"""
    with open(fpath, 'rb') as f:
        reader = csv.reader(f)
        dpx = float(reader.next()[0])
        dmm = float(reader.next()[0])
    scale = dmm / dpx
    return scale

def reference_length(fpath, scale):
    """Reads ref_length.csv"""
    with open(fpath, 'rb') as f:
        reader = csv.reader(f)
        l0 = float(reader.next()[0]) * scale
    return l0

def image_time(imname):
    """Parse image time from image filename.

    The image filenames are assumed to be in the convention followed
    by the Elliott lab LabView camera capture VI:
    
    `cam#_index_timeinseconds.tiff`

    """
    pattern = r"(?<=cam0_)([0-9]+)_([0-9]+\.?[0-9]+)(\.[A-Za-z]+)"
    timecode = re.search(pattern, imname).group(2)
    return float(timecode)

def read_image_index(fpath):
    """Returns dictionary of image names for specific test events.

    """
    with open(fpath, 'rb') as csvfile:
        image_index = dict()
        for row in csv.reader(csvfile):
            image_index[row[0]] = row[1]
    return image_index

def get_image_list(fpath):
    """Returns list of images and which test phase they belong to.

    Inputs
    ------
    fpath : string
        Path for 'image_index.csv'

    Returns
    -------
    list of 2-element tuples
        A sorted list of (image_file_path, test_phase) with the test
        phase of each image defined from the `image_index.csv` file in
        the image directory.

    """
    # Read image_index.csv
    image_index = read_image_index(fpath)
    # Check for image directory existence
    imdir = os.path.dirname(fpath)
    if not os.path.isdir(imdir):
        raise Exception("Could not find image directory at " +
                        imdir)
    # Build the image list
    images = []
    for fname in os.listdir(imdir):
        if fname.endswith((".tiff", ".tif")):
            images.append(fname)
    images.sort()
    imlist = []
    curr_phase = 'extra'
    image_index = dict((k, images.index(image_index[k]))
                       for k, v in image_index.iteritems()
                       if v != 'NA')
    for i, fname in enumerate(images):
        if (i >= image_index['reference'] and 
            i < image_index['ramp_start']):
            curr_phase = 'preconditioning'
        elif (i >= image_index['ramp_start'] and
              i <= image_index['end']):
            curr_phase = 'ramp'
        else:
            curr_phase = 'extra'
        imlist.append((fname, curr_phase))
    return imlist

def move_extra(fpath):
    """Moves extra images to 'unneeded' folder.

    """
    imlist = get_image_list(fpath)
    imdir = os.path.dirname(fpath)
    undir = os.path.join(imdir, "unneeded")
    for fname, phase in imlist:
        if phase == 'extra':
            os.rename(os.path.join(imdir, fname),
                      os.path.join(undir, fname))

def make_vic2d_lists(imidx, mechcsv, interval=0):
    """List images for vic2d analysis.

    """
    imdir = os.path.dirname(imidx)
    imdict = read_image_index(imidx)
    stretch, imnames = zip(*image_strain(imdir, mechcsv))
    imlist = get_image_list(imidx)

    reftime = image_time(imdict['reference'])

    # Choose images for ramp
    selected_images = [imdict['reference'],
                       imdict['ramp_start']]
    t0 = image_time(imdict['ramp_start'])
    if imdict.get('rupture') is not None:
        last_image = imdict['rupture']
    else:
        last_image = imdict['end']
    t1 = image_time(last_image)
    y1 = stretch[imnames.index(last_image)]
    yt = 0
    for i, y in enumerate(stretch):
        t = image_time(imnames[i])
        if (t > t0 and t < t1 and
            (y - yt > interval or
             y1 - y < 0.01)):
            yt = y
            selected_images.append(imnames[i])
    # Write image list
    with open(os.path.join(imdir, 'images_ramp_subsample.txt'), 'w') as f:
        for fname in selected_images:
            f.write(fname + '\n')

if __name__ == "__main__":
    """Hides images that are unnecessary for vic2d."""
    fpath = os.path.abspath(sys.argv[1])
    if not os.path.isfile(fpath):
        raise Exception(sys.argv[1] + " is not a file or does not exist.")
    move_extra(fpath)
