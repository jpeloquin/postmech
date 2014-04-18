import csv
import numpy as np
from PIL import Image
import re, os
from os import path
import h5py
from collections import defaultdict
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    csvonly = (f for f in files if
               not f.startswith('.')
               and f.endswith('.csv'))
    abspaths = (path.abspath(path.join(directory, f)) for f in csvonly)
    return sorted(list(abspaths))

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
        for row in df.iterrows():
            r = row[1]
            try:
                strainfield[(r['y'], r['x'])] = r[field]
            except IndexError:
                print('Warning: x or y index in row {} '
                      'is blank.'.format(row[0]))
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
