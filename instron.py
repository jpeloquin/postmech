#!/usr/bin/env python

import argparse
import os
import csv
import numpy as np
import mechana

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
