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
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pygismo as gismo
from . import vic2d
from . import read
from .unit import ureg
from .images import decode_impath, image_id, image_time, list_images, read_image_index, tabulate_images
from .write import try_unlock_on_fail

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

    def __init__(self, ssfile=None, imagelist=None, dir_vic2d=None):
        """Read test data from a test file.

        """
        ## Defaults
        self.strainfields = None

        # Read stress and strain
        if ssfile is not None:
            data = pd.read_csv(ssfile)
            self.stress = data['Stress (Pa)'].values
            self.stretch = data['Stretch Ratio'].values
            self.time = data['Time (s)'].values
        # Read images
        self.imagepaths = imagelist
        if self.imagepaths is not None:
            imdir = os.path.dirname(self.imagepaths[0])
            self.tab_images = tabulate_images(imdir,
                vic2d_dir=dir_vic2d)
            # Image times
            self.imagetimes = self.tab_images['Time (s)']
            # Image lookup table
            imageids = [image_id(f)
                        for f in self.imagepaths]
            self.imagedict = dict(zip(imageids, self.imagepaths))

        # Read strain fields
        if dir_vic2d is not None:
            # Read strain components from Vic-2D csv files
            vic2dfiles = vic2d.listcsvs(dir_vic2d)

            ncpu = multiprocessing.cpu_count()
            p = multiprocessing.Pool(ncpu)
            self.strainfields = p.map(vic2d.read_strain_components,
                                      vic2dfiles)
            p.close()
            p.join()

            ## Ignore empty strainfields
            self.strainfields = [a for a in self.strainfields
                                 if len(a['exx']) != 0]

            csvnames = (os.path.basename(f) for f in vic2dfiles)
            fieldtimes = [image_time(nm)
                          for nm in csvnames]
            t = float(self.tab_images.iloc[0]['Timestamp (s)'])
            self.image_t0 = t - self.tab_images.iloc[0]['Time (s)']
            self.fieldtimes = np.array(fieldtimes) - self.image_t0

    @classmethod
    def from_json(cls, pth_json):
        """Initialize from a json test record.

        The record format is the same as that output by
        postmech.organize.write_test_file.

        """
        datadir = os.path.dirname(pth_json)
        with open(pth_json, 'r') as f:
            testdesc = json.load(f)
        pth_ss = os.path.join(datadir, testdesc['stress_strain_file'])
        imagelist = testdesc['images']
        # Make the paths absolute (they are relative to the json file
        # location)
        imagelist = [os.path.join(datadir, fp)
                     for fp in testdesc['images']]
        # Vic-2D folder
        vic2dfolder=None
        if not pd.isnull(testdesc['vic2d_folder']):
            vic2dfolder = os.path.join(datadir,
                                    testdesc['vic2d_folder'])
        ## Create object
        self = cls(ssfile=pth_ss, imagelist=imagelist,
                   dir_vic2d=vic2dfolder)
        return self

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
        r = self.tab_images.iloc[idx]
        imtime = self.imagetimes[idx]
        impath = self.imagepaths[idx]
        imname = "cam{}_{}_{:.3f}".format(r['Camera ID'],
                                          r['Frame ID'],
                                          r['Timestamp (s)'])
        image = mpimg.imread(impath)

        mdata = {}
        mdata['time'] = imtime # this is based on the tensile tester
                               # clock
        mdata['name'] = imname

        return image, mdata

def imindex_lookup(csvpath, key):
    """Returns image name from image index.

    If `imagename` is not in the index, returns None.

    """
    with open(csvpath, 'r') as csvfile:
        for row in csv.reader(csvfile):
            if row[0] == key:
                return row[1]
    return None

def first_crossing(v, threshold, to='right'):
    """Find point in vector ≥ a threshold.

    to : 'left' or 'right'
        The direction in which the search moves.  'right' means start
        with the first timepoint and search by increasing timepoints;
        'left' means start with the last timepoint and search by
        decreasing timepoints.

    """
    sign_from_dir = {'left': -1,
                     'right': 1}
    direction = sign_from_dir[to]
    if direction == -1 and v[-1] >= threshold:
        # Last point already exceeds threshold
        idx = None
    elif direction == 1 and v[0] >= threshold:
        # First point already exceeds threshold
        idx = None
    else:
        idx = next(i for i, x in zip(np.arange(len(v))[::direction],
                                     v[::direction])
                   if x > threshold)
    return idx

def key_stress_pts(test):
    """Find image frames corresponding to key stress values.

    p := path to mechanical data csv table

    Points calculated:
    - Peak (max) stress
    - 1%, 2%, and 5% of peak stress, counting backward

    """
    # Get mechanical data
    df = pd.read_csv(test.stress_strain_file)
    # Get image data
    imlist = [os.path.basename(s)
              for s in list_images(test.image_dir)]
    imindex = read_image_index(
        os.path.join(test.image_measurements_dir, "image_index.csv"))
    imtime0 = image_time(imindex['ref_time'])
    with open(os.path.join(test.image_measurements_dir, "ref_time.csv")) as f:
        reader = csv.reader(f)
        reftime = float(reader.__next__()[0])
    d = imtime0 - reftime
    imtimes = [image_time(nm) - d for nm in imlist]

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

        idx = first_crossing(df['Stress (Pa)'].values,
                             threshold=f * peak_stress,
                             to='left')
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
    fpath = os.path.join(test.image_measurements_dir, "image_index.csv")
    for k in out:
        imindex[k] = out[k]
    with open(fpath, 'w', newline='') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,
                               lineterminator=os.linesep)
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
    fout = os.path.join(test.test_dir, "key_stress_pts_plot.svg")
    f = lambda: fig.savefig(fout)
    try_unlock_on_fail(f, fout)

    plt.close(fig)

def tabulate_stress_strain(spcdir, data, areapath, lengthpath,
                           imdir = None):
    """Calculate stress strain and make table.

    """
    if imdir is None:
        imdir = os.path.join(spcdir, "images")

    # Read area
    area = read.measurement_csv(areapath)
    area = area.to('m**2')

    # Read reference length
    ref_length = read.measurement_csv(lengthpath)
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

def xlim_strain(sstable):
    """Return useful axis limits for strain.

    """
    x0 = sstable['Stretch Ratio'].min()
    thresh = 0.01 * sstable['Stress (Pa)'].max()
    if sstable['Stress (Pa)'].iloc[-1] > thresh:
        x1 = sstable['Stretch Ratio'].max()
    else:
        idx = next(i for i in sstable.index[::-1]
                   if sstable['Stress (Pa)'][i] > thresh)
        x1 = sstable['Stretch Ratio'][idx]
    return (x0, x1)

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
    xlim = xlim_strain(sstable)
    ax.set_xlim(*xlim)

    # Re-layout figure
    fig.tight_layout()
    plt.close(fig)

    return fig

def ramp_data(test):
    """Truncate a stress-strain file to only include the ramp.

    """
    pth = os.path.join(test.test_dir, 'stress_strain.csv')
    data = pd.read_csv(pth)

    imindex = read_image_index(os.path.join(test.image_measurements_dir,
                                            'image_index.csv'))
    frame0 = decode_impath(imindex['ramp_start'])['Frame ID']
    image_tab = tabulate_images(test.image_dir,
                                test.stress_strain_file,
                                test.vic2d_dir)
    t0 = image_tab.loc[image_tab['Frame ID'] == frame0]['time (s)'].values[0]
    out = data[data['Time (s)'] >= t0]
    return out

def calculate_area(tab):
    areas = gismo.area_by_pass(tab)
    area = areas.mean() * ureg('mm**2')
    area = area.plus_minus(areas.std())
    area = area.to('m**2')
    return area
