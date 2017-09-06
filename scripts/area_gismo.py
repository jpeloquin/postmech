#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process gismo area output.

Run `area_gismo.py --help` from the command line for usage.

"""

import argparse
import os
import sys
import csv
import numpy as np

s = ("Calculate the average area from gismo_area.txt, "
     "which is assumed to have units of mm^2.")
parser = argparse.ArgumentParser(description=s)
parser.add_argument("infile", type=argparse.FileType('r'),
                    help="Path to gismo_area.txt")
parser.add_argument("outfile", nargs='?', type=argparse.FileType('w'),
                    default=sys.stdout,
                    help="Path to output file (default = stdout)")
args = parser.parse_args()
reader = csv.reader(args.infile, delimiter="\t")
# Skip header
for i in (0,1):
    next(reader)
# Read area from the 2D interpolation column
area = []
for row in reader:
    area.append(float(row[1]))
area = np.array(area) * 10.0**-6 ## mm² to m²
m = np.mean(area);
sd = np.std(area);
args.outfile.write(str(m) + "," + str(sd) + ',"m^2"\n')
