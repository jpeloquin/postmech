import csv
import numpy as np
from scipy.sparse import coo_matrix
from PIL import Image
import re, os
from os import path
import h5py
from collections import defaultdict
import pandas as pd
import h5py
import hashlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mechana
from mechana.images import name2key, image_strain

# Register diverging colormap

n = 256 # desired number of intensity levels
cdict_div = {'red': ((0, 0, 0.0941),
                     (0.1, 0.2745, 0.2745),
                     (0.2, 0.4275, 0.4275),
                     (0.3, 0.6275, 0.6275),
                     (0.4, 0.8118, 0.8118),
                     (0.5, 0.9451, 0.9451),
                     (0.6, 0.9569, 0.9569),
                     (0.7, 0.9725, 0.9725),
                     (0.8, 0.8824, 0.8824),
                     (0.9, 0.7333, 0.7333),
                     (1, 0.5647, 1)),
             'green': ((0.0, 0, 0.3098),
                       (0.1, 0.3882, 0.3882),
                       (0.2, 0.6000, 0.6000),
                       (0.3, 0.7451, 0.7451),
                       (0.4, 0.8863, 0.8863),
                       (0.5, 0.9569, 0.9569),
                       (0.6, 0.8549, 0.8549),
                       (0.7, 0.7216, 0.7216),
                       (0.8, 0.5725, 0.5725),
                       (0.9, 0.4706, 0.4706),
                       (1, 0.3922, 0)),
             'blue': ((0.0, 0, 0.6353),
                      (0.1, 0.6824, 0.6824),
                      (0.2, 0.8078, 0.8078),
                      (0.3, 0.8824, 0.8824),
                      (0.4, 0.9412, 0.9412),
                      (0.5, 0.9608, 0.9608),
                      (0.6, 0.7843, 0.7843),
                      (0.7, 0.5451, 0.5451),
                      (0.8, 0.2549, 0.2549),
                      (0.9, 0.2118, 0.2118),
                      (1, 0.1725, 1))}
matplotlib.cm.register_cmap(name="lab_diverging",
                            data=cdict_div, lut=256)

def listcsvs(directory):
    """List csv files in a directory.

    """
    files = sorted(os.listdir(directory))
    pattern = r'cam[0-9]_[0-9]+_[0-9]+.[0-9]{3}.csv'
    files = [f for f in files
             if re.match(pattern, f) is not None]
    return sorted(list(files))

def readv2dcsv(f):
    df = pd.read_csv(f, skipinitialspace=True).dropna(how='all')
    return df

class Vic2DDataset:
    """A class to represent a set of Vic-2D data files.

    """
    h5store = None
    keys = []
    hashes = []
    csvpaths = []

    def __init__(self, vicdir):
        # The class uses an hdf5 file as the data storage location
        h5path = os.path.join(vicdir, 'data.h5')
        if not os.path.exists(h5path):
            print h5path + ' does not exist; building.'
            hdf5ify(vicdir)
            self.h5store = h5py.File(h5path, 'r')
        else:
            self.load_h5(h5path)
            # Make sure the h5 file is up to date
            uptodate = True
            csvfiles = [os.path.join(vicdir, f) for f in listcsvs(vicdir)]
            csvhashes = [hashfile(f) for f in csvfiles]
            keys = [mechana.images.name2key(f) for f in csvfiles]
            h5hashdict = dict(zip(self.keys, self.hashes))
            csvhashdict = dict(zip(keys, csvhashes))
            allkeys = set(keys + self.keys.value.tolist())
            for k in allkeys:
                if k not in self.keys or k not in keys:
                    # a frame exists in only one location
                    uptodate = False
                if h5hashdict[k] != csvhashdict[k]:
                    # data is out of sync
                    uptodate = False
            if not uptodate:
                print h5path + ' not up to date; rebuilding.'
                self.h5store.close()
                os.remove(h5path)
                hdf5ify(vicdir)
                self.load_h5(h5path)

    def load_h5(self, h5path):
        self.h5store = h5py.File(h5path, 'r')
        self.keys = self.h5store["keys"]
        self.hashes = self.h5store["csvhashes"]
        self.csvpaths = self.h5store["csvpaths"]

    def __getitem__(self, key):
        group = self.h5store[key]
        fields = group.keys()
        columns = [group[k].value for k in fields]
        df = pd.DataFrame.from_dict(dict(zip(fields, columns)))
        return df

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        """Close the h5 file on object destruction.

        """
        self.h5store.close()

def hashfile(fpath):
    with open(fpath, 'rb') as f:
        fhash = hashlib.sha1(f.read()).hexdigest()
    return fhash

def hdf5ify(fdir, h5path=None):
    """Read csv files and store them in an hdf5 file.

    Storing the data in a binary format is intended to permit faster
    reads and interoperability between sofware packages.

    """
    savecolumns = ['x', 'y', 'u', 'v', 'exx', 'eyy', 'exy']

    # Define output path
    if h5path is None:
        h5path = os.path.join(fdir, 'data.h5')
    # Create list of vic2d csv files in the directory
    csvfiles = [os.path.join(fdir, f) for f in listcsvs(fdir)]
    key = [name2key(f) for f in csvfiles]
    fhashes = [hashfile(f) for f in csvfiles]
    hashdict = dict(zip(key, fhashes))
    fpdict = dict(zip(key, csvfiles))
    mdata = pd.DataFrame.from_dict({'key': key,
                                    'hash': fhashes,
                                    'csvpath': csvfiles})
    with h5py.File(h5path, 'w') as h5store:
        h5store.create_dataset("keys", data=key)
        h5store.create_dataset("csvpaths",
            data=[os.path.basename(f) for f in csvfiles])
        h5store.create_dataset("csvhashes", data=fhashes)
        for k, fp in zip(key, csvfiles):
            df = readv2dcsv(fp)
            grp = h5store.create_group(k)
            for c in savecolumns:
                try:
                    grp.create_dataset(c, data=df[c])
                except KeyError:
                    print ('Error reading file: ' + fp)
                    raise

def summarize_vic2d(vicdir, imdir):
    """Calculate summary statistics for Vic-2D data.

    Note: If the Vic-2D data were exported including blank regions,
    you will find many, many zeros in the data.

    """
    with open(path.join(imdir, 'mechdata_path.txt')) as f:
        mechpath = path.join(imdir, (f.read().strip()))
    imstrain = dict(image_strain(imdir, mechpath))
    fields = ['exx', 'eyy', 'exy']
    # Initialize output
    q05 = {k: [] for k in fields}
    q95 = {k: [] for k in fields}
    q50 = {k: [] for k in fields}
    strain = []
    keys = []
    vdset = mechana.vic2d.Vic2DDataset(vicdir)
    for k in vdset.keys:
        df = vdset[k]
        keys.append(k)
        strain.append(imstrain[k])
        for field in fields:
            q = np.percentile(df[field], [5, 50, 95])
            q05[field].append(q[0])
            q50[field].append(q[1])
            q95[field].append(q[2])
    out = {'key': keys,
           'strain': strain,
           'q05': q05,
           'median': q50,
           'q95': q95}
    return out

def strainimg(df, field):
    """Create a strain image from a list of values.

    """
    xmin = min(df['x'])
    xmax = max(df['x'])
    ymin = min(df['y'])
    ymax = max(df['y'])
    strainfield = np.empty((ymax+1, xmax+1))
    strainfield.fill(np.nan)
    x = df['x']
    y = df['y']
    v = df[field]
    ind = zip(y, x)
    strainfield[[y, x]] = v
    #for row in df.iterrows():
    #    r = row[1]
    #    try:
    #        strainfield[(r['y'], r['x'])] = r[field]
    #    except IndexError:
    #    print('Warning: x or y index in row {} '
    #    'is blank.'.format(row[0]))
    return strainfield

def plot_strains(csvpath, figpath):
    """Plot strain from a Vic-2D .csv file.

    """
    df = pd.read_csv(csvpath, skipinitialspace=True)

    # Find extent of region that has values
    xmin = min(df['x'])
    xmax = max(df['x'])
    ymin = min(df['y'])
    ymax = max(df['y'])

    def strainimg(df, field):
        strainfield = np.empty((ymax+1, xmax+1))
        strainfield.fill(np.nan)
        x = df['x']
        y = df['y']
        v = df[field]
        ind = zip(y, x)
        strainfield[[y, x]] = v
        #for i, j, val in zip(y, x, v):
        #    strainfield[i, j] = val
        return strainfield

    ## Initialize figure
    fig = plt.figure(figsize=(6.0, 2.5), dpi=300, facecolor='w')
    ax1 = fig.add_subplot(131, aspect='equal')
    ax2 = fig.add_subplot(132, aspect='equal')
    ax3 = fig.add_subplot(133, aspect='equal')
    axes = [ax1, ax2, ax3]
    matplotlib.rcParams.update({'font.size': 7})

    ## Add the three strain plots
    fields = ['exx', 'eyy', 'exy']
    ctitles = ['$e_{xx}$', '$e_{yy}$','$e_{xy}$']
    for i, field in enumerate(fields):
        ## Plot strain image
        im = strainimg(df, field)
        ax = axes[i]
        cmin, cmax = np.percentile(df[field].values, [5, 95])
        implot = ax.imshow(im, cmap="lab_diverging",
                           vmin=cmin, vmax=cmax)

        ## Format axis
        ax.axis('off')
        ax.axis((xmin, xmax, ymin, ymax))
        ax.invert_yaxis()

        ## Add colorbar
        cbar = fig.colorbar(implot,
                            orientation='horizontal',
                            ticks=matplotlib.ticker.MaxNLocator(nbins=6),
                            ax=ax)
        cbar.set_label(ctitles[i], size=10)

        ## Set colorbar limits
        clim = max(abs(cmin), abs(cmax))
        cbar.set_clim((-clim, clim))

    ## Format figure
    plt.tight_layout()

    return fig
