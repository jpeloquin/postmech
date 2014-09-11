# -*- coding: utf-8 -*-
"""Functions for analyzing test data.

Calculating metrics & deriving meaning.

"""
import os, json, csv

import pandas as pd
import matplotlib.pyplot as plt

import mechana
from mechana.unit import ureg

def label_unit(text):
    """Split a label (unit) string into parts.

    """
    pass

def key_stress_pts(fpath, imdir=None):
    """Find image frames corresponding to key stress values.

    Points calculated:
    - Peak (max) stress
    - 5% of peak stress, counting backward

    """
    # Get paths
    dirname = os.path.dirname(os.path.abspath(fpath))
    # Get mechanical data
    df = pd.read_csv(fpath)
    # Get image data
    if imdir is None:
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
        res_stress = f * peak_stress

        # find index of first stress value, from right, that exceeds
        # the threshold
        if df['Stress (Pa)'].iget(-1) > res_stress:
            # Last stress value already exceeds threshold
            idx = None
        else:
            idx = next(i for i in df.index[::-1]
                       if df['Stress (Pa)'][i] > res_stress)
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

def stress_strain(spcdir, mechpath, areapath, lengthpath,
                  notchpath=None, widthpath=None,
                  fmt='Bluehill', imdir=None):

    if imdir is None:
        imdir = os.path.join(spcdir, "images")

    # Read tensile test data
    if fmt == 'Wintest':
        data = mechana.read.bose_data(mechpath)
    elif fmt == 'Bluehill':
        data = mechana.read.instron_data(mechpath)
    else:
        raise Exception("Specified format '"
                        + fmt + "' is not supported.")

    # Read area
    area = mechana.read.measurement_csv(areapath)
    area = area[0].to('m**2').magnitude

    # Read reference length
    ref_length = mechana.read.measurement_csv(lengthpath)
    ref_length = ref_length[0].to('m').magnitude

    # If notched, reduce effective area
    if notchpath:
        fpath = notchpath
        with open(fpath, 'rb') as f:
            reader = csv.reader(f)
            a = float(reader.next()[0])
        fpath = widthpath
        with open(fpath, 'rb') as f:
            reader = csv.reader(f)
            w = float(reader.next()[0])
        area = area * (1 - a / w)

    # Calculate stretch and stress
    # Stretch
    length = ref_length + (data['Position (m)']
                           - data['Position (m)'][0])
    data['Stretch Ratio'] = length / ref_length
    # Stress
    data['Stress (Pa)'] = data['Load (N)'] / area

    # Generate a plot
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    # converting to MPa
    ax.plot(data["Stretch Ratio"],
            data["Stress (Pa)"] / 1e6,
            color='k')
    ax.set_xlabel("Stretch")
    ax.set_ylabel("Stress (MPa)")
    fig.tight_layout()
    fig.savefig(os.path.join(spcdir, "stress_strain.svg"))

    # Write output
    outpath = os.path.join(spcdir, "stress_strain.csv")
    data.to_csv(outpath, index=False)
