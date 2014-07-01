# -*- coding: utf-8 -*-
"""Functions for analyzing test data.

Calculating metrics & deriving meaning.

"""
import os, json, csv
import pandas as pd
import mechana

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
    imlist = mechana.images.list_images(imdir)
    imindex = mechana.images.read_image_index(
        os.path.join(imdir, "image_index.csv"))
    imtime0 = mechana.images.image_time(imindex['ref_time'])
    with open(os.path.join(imdir, "ref_time.csv")) as f:
        reader = csv.reader(f)
        reftime = float(reader.next()[0])
    d = imtime0 - reftime
    imtimes = [mechana.images.image_time(nm) - d 
                  for nm in imlist]

    # Helper function
    def closest_idx(target, values, count_from='right'):
        """Find index of closest value by simple counting.

        The closest value is found by checking each value in sequence
        until the target threshold is crossed for the first time.
        (The function determines whether the sequence is increasing
        nor decreasing based on whether the first value is greater
        than or less than the target value.)

        Values can be checked from left to right or right to left.
        Any value that would be subsequently checked is ignored.

        """
        # Setup
        if count_from == 'right':
            idx = len(values) - 1
            increment = -1
        elif count_from == 'left':
            idx = 0
            increment = 1
        else:
            raise Exception("count_from must be 'left' or 'right'")
        # Run the loop
        v = values[idx]
        # Figure out if we're counting up or down
        if values[idx] < target:
            # values should increase
            while v < target:
                idx = idx + increment
                v = values[idx]
        else:
            # values should decrease
            while v > target:
                idx = idx + increment
                v = values[idx]
        idx = idx - increment
        # Check if result outside of bounds
        if idx < 0:
            raise ValueError("The sequence does not include a point to the left of the target value.")
        if idx > (len(values) - 1):
            raise ValueError("The sequence does not include a point to the right of the target value.")
        return idx

    # Allocate output
    out = {}

    # Peak stress
    peak_stress = df['Stress (Pa)'].max()
    idx_peak = df.idxmax()['Stress (Pa)']
    idx = closest_idx(df['Time (s)'][idx_peak],
                      imtimes, count_from='right')
    imname = os.path.basename(imlist[idx])
    out['peak stress'] = imname

    # Residual stress
    resfrac = [0.01, 0.02, 0.05]
    idx_res = []
    for f in resfrac:
        res_stress = f * peak_stress
        idx_res.append(closest_idx(res_stress, df['Stress (Pa)'],
                                   count_from='right'))
        idx = closest_idx(df['Time (s)'][idx_res[-1]],
                          imtimes, count_from='right')
        imname = os.path.basename(imlist[idx])
        key = '{}% residual strength'.format(int(round(f * 100)))
        out[key] = imname

    # Write frames to image index
    fpath = os.path.join(imdir, "image_index.csv")
    for k in out:
        imindex[k] = out[k]
    with open(fpath, 'w') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for k in imindex:
            csvwriter.writerow([k, imindex[k]])

    # Plot the key points
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,3), dpi=600)
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
