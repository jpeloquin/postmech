# -*- coding: utf-8 -*-
"""Functions for analyzing test data.

Calculating metrics & deriving meaning.

"""
import csv
import json
import os
import re
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import mechana
from mechana.unit import ureg
from mechana.vic2d import read_strain_components

class MechanicalTest(object):
    """Mechanical test data.

    """
    # Mechanical data (arrays have 1:1 mapping)
    # stretch = None
    # stress = None
    # time = None

    # imagepaths = None
    # imagenames = None
    # image_t0 = None

    # strainfields = None # list of monochrome uint8 images
    # strainfields_argb = None # list of rgba images
    # fieldtimes = None

    def __init__(self, jsonfile):
        """Read test data from a test file.

        """
        datadir = os.path.dirname(jsonfile)
        with open(jsonfile, 'rb') as f:
            testdesc = json.load(f)
        # Read stress and strain
        ssfile = os.path.join(datadir, testdesc['stress_strain_file'])
        if ssfile is not None:
            data = pd.read_csv(ssfile)
            self.stress = data['Stress (Pa)'].values
            self.stretch = data['Stretch Ratio'].values
            self.time = data['Time (s)'].values
        # Read images
        imagelist = testdesc['images']
        if imagelist is not None:
            # Image list
            # Make the paths relative to the json file
            imagelist = [os.path.join(datadir, fp)
                         for fp in testdesc['images']]
            self.imagepaths = imagelist
            # Image times
            self.imagenames = [os.path.basename(f) for f in imagelist]
            imagetimes = [mechana.images.image_time(nm)
                          for nm in self.imagenames]
            # Guess at image index path
            imindex = mechana.images.read_image_index(os.path.join(datadir, 'images', 'image_index.csv'))
            reftime = mechana.read.measurement_csv(os.path.join(datadir, 'images', 'ref_time.csv'))
            reftime = reftime.nominal_value
            self.image_t0 = mechana.images.image_time(imindex['ref_time']) - reftime
            self.imagetimes = np.array(imagetimes) - self.image_t0
            # Image lookup table
            imageids = [mechana.images.image_id(f)
                        for f in imagelist]
            self.imagedict = dict(zip(imageids, imagelist))
        else:
            self.imagepaths = None
        # Read strain fields
        vic2dfolder = testdesc['vic2d_folder']
        if vic2dfolder is not None:
            # make vic2d folder relative to json file
            vic2dfolder = os.path.join(datadir, vic2dfolder)
            if self.image_t0 is None:
                raise Exception("Cannot match Vic-2D times "
                                "to mechanical test data "
                                "without an offset defined "
                                "for the image time codes.")

            # Read strain components from Vic-2D csv files
            vic2dfiles = mechana.vic2d.listcsvs(vic2dfolder)

            ncpu = multiprocessing.cpu_count()
            p = multiprocessing.Pool(ncpu)
            self.strainfields = p.map(read_strain_components, vic2dfiles)

            csvnames = (os.path.basename(f) for f in vic2dfiles)
            fieldtimes = [mechana.images.image_time(nm)
                          for nm in csvnames]
            self.fieldtimes = np.array(fieldtimes) - self.image_t0

    def stress_at(self, t):
        """Return stress at time t.

        """
        return np.interp(t, self.time, self.stress)

    def stretch_at(self, t):
        """Return stretch at time t.

        """
        return np.interp(t, self.time, self.stretch)

    def image_at(self, t):
        """Return frame (and metadata) at time t.

        """
        idx = np.argmin(np.abs(self.imagetimes - t))
        imtime = self.imagetimes[idx]
        impath = self.imagepaths[idx]
        imname = self.imagenames[idx]
        image = mpimg.imread(impath)

        mdata = {}
        mdata['time'] = imtime # this is based on the tensile tester
                               # clock
        mdata['name'] = imname

        return image, mdata


def imtime(imlist):
    """Get timecode from list of test image filenames.

    The image filenames are assumed to be in the convention followed
    by the Elliott lab LabView camera capture VI:

    `cam#_index_timeinseconds.tiff`

    """
    if type(imlist) is str:
        imlist = [imlist]
    pattern = r"(?<=cam0_)([0-9]+)_([0-9]+\.?[0-9]+)(\.[A-Za-z]+)"
    t = []
    for s in imlist:
        timecode = re.search(pattern, s).group(2)
        t.append(float(timecode))
    return t

def imindex_lookup(csvpath, key):
    """Returns image name from image index.

    If `imagename` is not in the index, returns None.

    """
    with open(csvpath, 'r') as csvfile:
        for row in csv.reader(csvfile):
            if row[0] == key:
                return row[1]
    return None

def label_unit(text):
    """Split a label (unit) string into parts.

    """
    pass

def stress_crossing(tab, thresh, direction='left'):
    """Find stress-strain point crossing a stress threshold.

    """
    # find index of first stress value, from right, that exceeds
    # the threshold
    if tab['Stress (Pa)'].iget(-1) > thresh:
        # Last stress value already exceeds threshold
        idx = None
    else:
        sign_from_dir = {'left': -1,
                         'right': 1}
        idx = next(i for i in tab.index[::sign_from_dir[direction]]
                   if tab['Stress (Pa)'][i] > thresh)

    return idx


def key_stress_pts(fpath, imdir=None):
    """Find image frames corresponding to key stress values.

    Points calculated:
    - Peak (max) stress
    - 1%, 2%, and 5% of peak stress, counting backward

    """
    # Get paths
    dirname = os.path.dirname(os.path.abspath(fpath))
    # Get mechanical data
    df = pd.read_csv(fpath)
    # Get image data
    if pd.isnull(imdir):
        imdir = os.path.join(dirname, "images")
    imlist = [os.path.basename(s)
              for s in mechana.images.list_images(imdir)]
    imindex = mechana.images.read_image_index(
        os.path.join(imdir, "image_index.csv"))
    imtime0 = mechana.images.image_time(imindex['ref_time'])
    with open(os.path.join(imdir, "ref_time.csv")) as f:
        reader = csv.reader(f)
        reftime = float(reader.next()[0])
    d = imtime0 - reftime
    imtimes = [mechana.images.image_time(nm) - d
               for nm in imlist]

    # Allocate output
    out = {}

    # Peak stress
    peak_stress = df['Stress (Pa)'].max()
    idx_peak = df.idxmax()['Stress (Pa)']
    out['peak stress'] = next(nm for nm, t in zip(imlist, imtimes)
                              if t > df['Time (s)'][idx_peak])

    # Residual stress
    resfrac = [0.01, 0.02, 0.05][::-1]
    # The list is reversed because, when the residual strength points
    # are plotted, the smaller residual strength points may not exist
    # on the curve.  Making them come last means their absence won't
    # affect the assignment of colors for the other points.
    idx_res = []
    for f in resfrac:
        key = '{}% residual strength'.format(int(round(f * 100)))

        idx = stress_crossing(df, thresh = f * peak_stress)
        idx_res.append(idx)

        # find index of corresponding image (nearest following time)
        if idx_res[-1] is not None:
            tc = df['Time (s)'][idx_res[-1]]
            imname = next((nm for t, nm in zip(imtimes, imlist)
                           if t > tc),
                          None)
            out[key] = imname
        else:
            out[key] = None

    # Write frames to image index
    fpath = os.path.join(imdir, "image_index.csv")
    for k in out:
        imindex[k] = out[k]
    with open(fpath, 'wb') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for k in imindex:
            v = imindex[k]
            if v is None:
                v = "NA"
            csvwriter.writerow([k, v])

    # Plot the key points
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ydiv = 1e6
    ax.plot(df['Stretch Ratio'], df['Stress (Pa)'] / ydiv,
            color='k')
    plt.tick_params(axis='both', which='major', labelsize=10)
    ln_peak, = ax.plot(df.loc[idx_peak]['Stretch Ratio'],
                       df.loc[idx_peak]['Stress (Pa)'] / ydiv,
                       marker='o', markersize=6,
                       linestyle='None')
    # legend
    legend_labels = ['Peak stress']
    lns_res = []
    for i, f in enumerate(resfrac):
        if idx_res[i] is not None:
            ln_res, = ax.plot(df.loc[idx_res[i]]['Stretch Ratio'],
                              df.loc[idx_res[i]]['Stress (Pa)'] / ydiv,
                              marker='o', markersize=6,
                              linestyle='None')
            lns_res.append(ln_res)
            legend_labels.append("{}% strength".format(
                int(round(f * 100))))
    ax.legend([ln_peak] + lns_res, legend_labels,
              loc='upper right', prop={'size': 10},
              numpoints=1)
    # axis formatting
    ax.set_xlabel("Stretch ratio")
    ax.set_ylabel("Stress (MPa)")
    fig.tight_layout()
    fout = os.path.join(dirname, "key_stress_pts_plot.svg")
    fig.savefig(fout)

    plt.close(fig)

def tabulate_stress_strain(spcdir, data, areapath, lengthpath,
                           imdir = None):
    """Calculate stress strain and make table.

    """
    if imdir is None:
        imdir = os.path.join(spcdir, "images")

    # Read area
    area = mechana.read.measurement_csv(areapath)
    area = area.to('m**2')

    # Read reference length
    ref_length = mechana.read.measurement_csv(lengthpath)
    ref_length = ref_length.to('m')

    # Calculate stretch and stress
    # Stretch
    # Uncertainty is discarded here
    length = (ref_length.value.magnitude
              + (data['Position (m)']
                 - data['Position (m)'][0]))
    data['Stretch Ratio'] = length / ref_length.value.magnitude
    # Stress
    data['Stress (Pa)'] = data['Load (N)'] / area.value.magnitude

    return data

def plot_stress_strain(sstable):
    """Plot stress vs. strain for a tensile test.

    """
    # Generate a plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    # Convert stress to MPa
    ax.plot(sstable["Stretch Ratio"],
            sstable["Stress (Pa)"] / 1e6,
            color='k')
    ax.set_xlabel("Stretch")
    ax.set_ylabel("Stress (MPa)")

    # Set x axis limits
    x0 = sstable['Stretch Ratio'].min()
    thresh = 0.01 * sstable['Stress (Pa)'].max()
    if sstable['Stress (Pa)'].iget(-1) > thresh:
        x1 = sstable['Stretch Ratio'].max()
    else:
        idx = next(i for i in sstable.index[::-1]
                   if sstable['Stress (Pa)'][i] > thresh)
        x1 = sstable['Stretch Ratio'][idx]
    ax.set_xlim(x0, x1)

    # Re-layout figure
    fig.tight_layout()
    plt.close(fig)

    return fig
