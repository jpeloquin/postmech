#!/usr/bin/env python

import argparse
import os
import csv
import numpy as np
import mechana

def read_instron_csv(fpath):
    """Read data from an Instron csv file.

    The function expects to find time, extension, and load data.  It
    assumes that time is the first column.

    Outputs
    -------
    time, extension, load : numpy array

    """
    t = []
    d = []
    p = []    
    with open(fpath, 'rb') as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for i in range(6): # skip header
            reader.next()
        header = reader.next() # read column names
        # Check that we arrived at the right row
        assert header[0] == 'Time'
        # Find Load and Extension columns
        dind = header.index('Extension')
        pind = header.index('Load')
        units = reader.next() # read units
        for row in reader:
            t.append(float(row[0]))
            d.append(float(row[dind]))
            p.append(float(row[pind]))
    t = np.array(t)
    d = np.array(d)
    p = np.array(p)
    return t, d, p

def stretch_ratio(d, l0):
    return (d + l0) / l0

def stress(p, a):
    """Calculate 1st Piola-Kirchoff stress.

    Parameters
    ----------
    p : 1-D array or list
       Load in Newtons
    a : numeric
       Area in meters

    """
    return np.array(p) / a

if __name__ == "__main__":
    # Arguments
    s = 'Calculate strain from extension data recorded by Bluehill (Instron).'
    parser = argparse.ArgumentParser(description=s)
    parser.add_argument('input', 
                        help='Path of Instron Raw Data csv file.')
    parser.add_argument('--imdir',
                        help='Path of image directory.')
#    parser.add_argument('--scale',
#                       help='Path to csv file containg scale measurements')
#    parser.add_argument('--lref',
#                       help='Path to csv file containing the reference '
#                       'length (px).')
    parser.add_argument('--notch', action='store_true',
                        help='Flag.  If set, look for "notch_length.csv" and '
                        '"ref_width.csv".  The cross-sectional area will be adjusted '
                        'accordingly.')
    parser.add_argument('-o', '--output', dest='outfile',
                        default=None,
                        help='Path of output file.  The default is to '
                        'write the output file to the same directory as '
                        'the input file.')
    args = parser.parse_args()

    # File paths
    datadir = os.path.dirname(os.path.dirname(args.input))
    if args.imdir:
        imdir = args.imdir
    else:
        imdir = os.path.join(datadir, "images")
    # Calculations
    scale = mechana.images.image_scale(os.path.join(imdir,
                                                    "image_scale.csv"))
    # read reference length
    fpath = os.path.join(imdir, "ref_length.csv")
    with open(fpath, 'rb') as f:
        reader = csv.reader(f)
        l0 = float(reader.next()[0])
    t, d, p = read_instron_csv(args.input)
    y = stretch_ratio(d, l0)
    fpath = os.path.join(datadir, "area.csv")
    with open(fpath, 'r') as f:
        reader = csv.reader(f)
        area = float(reader.next()[0])
    # Adjust for notch
    if args.notch:
        fpath = os.path.join(imdir, "notch_length.csv")
        with open(fpath, 'rb') as f:
            reader = csv.reader(f)
            a = float(reader.next()[0])
        fpath = os.path.join(imdir, "ref_width.csv")
        with open(fpath, 'rb') as f:
            reader = csv.reader(f)
            w = float(reader.next()[0])
        area = area * (1 - a / w)
    elif os.path.exists(os.path.join(imdir, "notch_length.csv")):
        print("\nWarning: notch_length.csv exists, but the --notch "
              "flag was not specified.  Stress will be calculated "
              "as if the specimen is crack-free.\n")
    s = stress(p, area)
    # Write output
    if args.outfile:
        fpathout = args.outfile
    else:
        fpathout = os.path.join(datadir, "stress_strain.csv")
    with open(fpathout, 'wb') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"')
        writer.writerow(["Time (s)", "Stretch Ratio", "Stress (Pa)"])
        for v in zip(t, y, s):
            writer.writerow(v)
    print("Wrote output to " + fpathout)
    print("Max stress was {:.2e} Pa (units assumed)".format(max(s)))
