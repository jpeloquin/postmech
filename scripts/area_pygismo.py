#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize specimen area from pygismo table.

Run `area_pygismo.py --help` from the command line for usage.

"""

import argparse

import sys
import pygismo as gismo

from ..analysis import calculate_area
from ..write import measurement_csv

import numpy as np

s = (
    "Calculate the average area from gismo_labeled.csv, "
    "which is assumed to have units of mm^2."
)
parser = argparse.ArgumentParser(description=s)
parser.add_argument("infile", type=str, help="Path of gismo_labeled.csv")
parser.add_argument(
    "outfile",
    nargs="?",
    type=argparse.FileType("w"),
    default=sys.stdout,
    help="Path of output file (default = stdout)",
)
args = parser.parse_args()
data = gismo.read_labeled(args.infile)
area = calculate_area(data)
measurement_csv(area, args.outfile)
