# -*- coding: utf-8 -*-
"""Functions for rearranging and organizing test data.

"""
import os, json


def write_test_file(
    fout="test_data.json", ssfile="stress_strain.csv", vic2d_folder=None, images=None
):
    """Write a JSON file that gathers all the data for a test.

    The JSON file specifies:

    - Which images belong to the test
    - The file containing stress & strain data
    - The folder in which the Vic-2D data is stored

    Paths are defined relative to the directory in which the JSON file
    is stored.

    Default assumptions:

    - The first image in the list was taken approximately at the same
      time the stress & strain test data begins.

    """
    # Paths
    fout = os.path.abspath(fout)
    outdir = os.path.dirname(fout)
    # Initialize variables
    data = {}
    # Stress-strain data location
    data["stress_strain_file"] = os.path.relpath(ssfile, outdir)
    # Vic-2D data location
    if vic2d_folder is None:
        data["vic2d_folder"] = None
    else:
        data["vic2d_folder"] = os.path.relpath(vic2d_folder, outdir)
    # List of images
    if images is None:
        data["images"] = None
    else:
        images = [os.path.relpath(impath, outdir) for impath in images]
        data["images"] = images
    # Write JSON
    with open(fout, "w") as f:
        json.dump(data, f)
