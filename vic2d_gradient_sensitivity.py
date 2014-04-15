# -*- coding: utf-8 -*-
# Part of the mechana package
from mechana.images import image_strain
import numpy as np
import os
import os.path as path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import matplotlib.cm
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

def summarize_vic2d(files, imindex):
    """Calculate summary statistics for Vic-2D data.

    Note: If the Vic-2D data were exported including blank regions,
    you will find many, many zeros in the data.

    """
    imdir = path.dirname(imindex)
    with open(path.join(imdir, 'mechdata_path.txt')) as f:
        mechpath = path.join(imdir, (f.read().strip()))
    imstrain = dict(image_strain(imdir, mechpath))
    usekeys = ['exx', 'eyy', 'exy']
    # Initialize output
    q05 = {k: [] for k in usekeys}
    q95 = {k: [] for k in usekeys}
    q50 = {k: [] for k in usekeys}
    strain = []
    names = []
    # Iterate over csv files
    for fp in files:
        df = pd.read_csv(fp, skipinitialspace=True)
        nm, ext = path.splitext(path.basename(fp))
        names.append(nm)
        strain.append(imstrain[nm + '.tiff'])
        for k in usekeys:
            q = np.percentile(df[k], [5, 50, 95])
            q05[k].append(q[0])
            q50[k].append(q[1])
            q95[k].append(q[2])
    out = {'names': names,
           'strain': strain,
           'q05': q05,
           'median': q50,
           'q95': q95}
    return out

def quant_compare(data1, data2, field, imindex,
                  title1=None, title2=None):
    """Calculate and differences in strain fields between two directories.

    """
    # Make sure lists match up
    if len([(a, b) for a, b 
            in zip(data1['names'], data2['names'])
            if a != b]) > 0:
        raise Exception('Datasets include different timepoints.')
    # Plot the differences
    matplotlib.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(4, 3), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Stretch Ratio (Grip-to-Grip)')
    ax.set_ylabel('Stretch Ratio (Texture Correlation)')
    lm_0 = plt.plot(data1['strain'], data1['median'][field],
                    'k-')[0]
    lq05_0 = plt.plot(data1['strain'], data1['q05'][field],
                      'k--')[0]
    lq95_0 = plt.plot(data1['strain'], data1['q95'][field],
                      'k--')[0]
    lm_1 = plt.plot(data2['strain'], data2['median'][field],
                    'r-')[0]
    lq05_1 = plt.plot(data2['strain'], data2['q05'][field],
                      'r--')[0]
    lq95_1 = plt.plot(data2['strain'], data2['q95'][field],
                      'r--')[0]
    ax = fig.gca()
    leg_data = plt.legend((lm_0, lm_1),
                          (title1, title2),
                          loc='upper left')
    return fig

def vis_compare(f1, f2, field, imindex, title1=None, title2=None,
                cbar_label=None):
    """Plot strain fields and their differences.

    """
    df1 = pd.read_csv(f1, skipinitialspace=True)
    df2 = pd.read_csv(f2, skipinitialspace=True)

    # Plot the differences
    nm, ext = path.splitext(path.basename(f1))
    imdir = path.dirname(imindex)
    impath = path.join(imdir, nm) + '.tiff'
    img = mpimg.imread(impath) # indexed y, x

    def strainimg(df):
        strainfield = np.empty(img.shape)
        strainfield.fill(np.nan)
        for row in df.iterrows():
            r = row[1]
            try:
                strainfield[(r['y'], r['x'])] = r[field]
            except IndexError:
                print('Warning: x or y index in row {} '
                      'is blank.'.format(row[0]))
        return strainfield
    
    im1 = strainimg(df1)
    im2 = strainimg(df2)
    imdiff = im1 - im2

    v = np.concatenate([df1[field].values, df2[field].values])
    cmax = np.percentile(v, 95)
    cmin = np.percentile(v, 5)
    if cmin < 0:
        cmin = -cmax

    # Find extent of region that has values
    xmin = min(min(df1['x']), min(df2['x']))
    xmax = max(max(df1['x']), max(df2['x']))
    ymin = min(min(df1['y']), min(df2['y']))
    ymax = max(max(df1['y']), max(df2['y']))

    matplotlib.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=(6.0, 2.5), dpi=300, facecolor='w')
    
    ax1 = fig.add_subplot(131, aspect='equal')
    fieldplot1 = plt.imshow(im1, cmap="lab_diverging")
    cbar = plt.colorbar(orientation='horizontal',
                        ticks=matplotlib.ticker.MaxNLocator(nbins=7))
    cbar.set_label(cbar_label, size=10)
    plt.clim((cmin, cmax))
    ax1.axis((xmin, xmax, ymin, ymax))
    ax1.axis('off')
    ax1.invert_yaxis()
    if title1 is not None:
        ax1.set_title('(A)' + title1)

    ax2 = fig.add_subplot(132, aspect='equal')
    fieldplot2 = plt.imshow(im2, cmap="lab_diverging")
    cbar = plt.colorbar(orientation='horizontal',
                        ticks=matplotlib.ticker.MaxNLocator(nbins=7))
    cbar.set_label(cbar_label, size=10)
    plt.clim((cmin, cmax))
    ax2.axis((xmin, xmax, ymin, ymax))
    ax2.axis('off')
    ax2.invert_yaxis()
    if title2 is not None:
        ax2.set_title('(B)' + title2)

    ax3 = fig.add_subplot(133, aspect='equal')
    diffplot = plt.imshow(imdiff, cmap="lab_diverging")
    cbar = plt.colorbar(orientation='horizontal',
                        ticks=matplotlib.ticker.MaxNLocator(nbins=7))
    cbar.set_label(cbar_label, size=10)
    plt.clim((-cmax, cmax))
    ax3.axis((xmin, xmax, ymin, ymax))
    ax3.axis('off')
    ax3.invert_yaxis()
    ax3.set_title(u'(C) Difference (A − B)')

    plt.tight_layout()

    return fig
