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

import numpy as np

import mechana
from mechana import instron
from mechana.unit import ureg

def image_id(fpath):
    """Convert image name into a unique id.

    """
    s = os.path.basename(fpath)
    pattern = r'(?P<key>cam[0-9]_[0-9]+)[0-9._A-Za-z]+(?:.tiff|.csv|.tif)?'
    m = re.search(pattern, s)
    return m.group('key')

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
    reftime = image_time(imdict["ref_time"])
    imtimes = [image_time(n) - reftime for n in imnames]
    scale = image_scale(os.path.join(imdir, 'image_scale.csv'))

    # Calculate reference length
    l0 = reference_length(os.path.join(imdir, 'ref_length.csv'), scale)

    # Load mechanical test data
    t, d, p = mechana.instron.read_instron_csv(mechcsv)
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
        for line in reader:
            unit = line[-1]
            if unit == "px":
                unit = ureg(unit)
                px_d = float(line[0]) * unit
                px_sd = float(line[1]) * unit
            else:
                unit = ureg(unit)
                d = float(line[0]) * unit
                sd = float(line[1]) * unit
    scale = d / px_d
    scale_sd = abs(scale) * ((sd / d)**2.0 + (px_sd / px_d)**2.0)**0.5
    return scale, scale_sd

def from_px(fpath, scale):
    """Reads ref_length.csv

    scale := (value, sd)

    """
    d, sd = mechana.read.measurement_csv(fpath)
    mm_d =  d * scale[0]
    mm_sd = abs(mm_d) * ((sd / d)**2.0 + (scale[1] / scale[0])**2.0)**0.5
    return mm_d, mm_sd

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

def list_images(directory):
    """List image files in a directory.

    """
    files = sorted(os.listdir(directory))
    pattern = r'cam0_[0-9]+_[0-9]+.[0-9]{3}.tiff'
    files = [f for f in files
             if re.match(pattern, f) is not None]
    files = [os.path.join(directory, f) for f in files]
    return sorted(list(files))

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
        if (i >= image_index['ref_time'] and
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
    fpath = os.path.abspath(fpath)
    imlist = get_image_list(fpath)
    imdir = os.path.dirname(fpath)
    undir = os.path.join(imdir, "unneeded")
    if not os.path.exists(undir):
        os.makedirs(undir)
    for fname, phase in imlist:
        if phase == 'extra':
            os.rename(os.path.join(imdir, fname),
                      os.path.join(undir, fname))

def make_vic2d_lists(fp, mechcsv, interval=0.01, highres=None,
                     fout='vic2d_list.csv',
                     zero_strain='zero_strain',
                     start='vic2d_start',
                     end='vic2d_end'):
    """List images for vic2d analysis.

    Inputs
    ------
    highres : tuple
        If an image has a stretch ratio >= highres[0] and <=
        highres[1], it will be included in the list even if it would
        be skipped according the interval setting.
    start : string
        The key in image_index.csv naming the first frame to be
        included in the list.
    end : string
        The key in image_index.csv naming the last frame to be
        included in the list.

    """
    # Calculate paths
    fp_imindex = os.path.abspath(fp)
    imdir = os.path.dirname(fp_imindex)
    fp_ss = os.path.abspath(mechcsv)

    imindex = read_image_index(fp_imindex)
    imlist = mechana.images.list_images(imdir)
    imnames, stretch = zip(*image_strain(imdir, mechcsv))

    selected_images = [imindex[zero_strain]]

    t_start = image_time(imindex[start])
    t_end = image_time(imindex[end])
    imname_start = imindex[start]
    imname_end = imindex[end]

    y_start = stretch[imnames.index(imname_start)]
    y_end = stretch[imnames.index(imname_end)]
    yt = 0
    for i, y in enumerate(stretch):
        t = image_time(imnames[i])
        if highres is not None:
            inhighres = (y >= highres[0] and y <= highres[1])
        else:
            inhighres = False
        if (t > t_start and t <= t_end
            and (y - yt > interval or inhighres)):
            yt = y
            selected_images.append(imnames[i])

    # Write image list
    with open(fout, 'w') as f:
        for nm in selected_images:
            f.write(nm + '\n')

if __name__ == "__main__":
    """Hides images that are unnecessary for vic2d."""
    fpath = os.path.abspath(sys.argv[1])
    if not os.path.isfile(fpath):
        raise Exception(sys.argv[1] + " is not a file or does not exist.")
    move_extra(fpath)
