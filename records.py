import json
import os
from os.path import join as pjoin
import logging
logger = logging.getLogger(__name__)
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .images import list_images

class Test:
    """Find and store paths of test data files from test records.

    """

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.test_dir = None
        self.stress_strain_file = None
        self.image_dir = None
        self.image_archive = None
        self.image_paths = []
        self.image_measurements_dir = None
        self.vic2d_dir = None
        self.vic2d_paths = []
        self.record = {}

    @classmethod
    def from_row(cls, project_dir, row):
        self = cls(project_dir)

        ## Save test record
        self.record = row

        ## Primary data folder for test
        if not pd.isnull(row['ramp to failure folder']):
            self.test_dir = os.path.join(project_dir, 'data',
                                         row['ramp to failure folder'])

        ## Images and image-derived measurements (for tests with images)
        if not pd.isnull(row['image directory']):
            p_images = os.path.join(project_dir, 'data', row['image directory'])

            # If the test uses a .zip image archive, inflate it and set
            # the image directory to the inflated directory.  We need
            # unzipped copies of the images.  Lots of code relies on the
            # images existing as individual files.
            if p_images.endswith('.zip'):
                self.image_archive = p_images
                self.image_dir = p_images[:-4]
                need_unzip = False
                if not os.path.exists(self.image_dir):
                    need_unzip = True
                    os.mkdir(self.image_dir)
                elif len(list_images(self.image_dir)) == 0:
                    need_unzip = True
                if need_unzip:
                    logger.info("Unzipping {}".format(self.image_archive))
                    ZipFile(self.image_archive).extractall(self.image_dir)
            else:
                self.image_dir = p_images

            # List the images in the image directory
            self.image_paths = list_images(self.image_dir)

            # Find the directory with image-derived measurements
            d = pjoin(self.test_dir, 'image_measurements')
            if os.path.exists(d):
                self.image_measurements_dir = d

        ## Vic-2d directory
        if not pd.isnull(row['vic-2d export folder']):
            self.vic2d_dir = os.path.join(project_dir,
                                          row['vic-2d export folder'])

        ## Stress and strain data file
        if self.test_dir is not None:
            self.stress_strain_file = os.path.join(self.test_dir,
                                                   'stress_strain.csv')
            if not os.path.exists(self.stress_strain_file):
                self.stress_strain_file = None
                print("Warning: {} test {} has no stress_strain.csv "
                      "file.".format(row['specimen id'], row['test id']))

        return self

    def __getitem__(self, key):
        return self.record[key]

def meniscus_cr_basis(s_c, s_r):
    """Return parsed circumferential-radial basis vectors from table.

    Arguments
    ---------
    s_c := The circumferential basis vector entry, posterior to anterior.

    s_r := The radial basis vector entry, inner to outer.

    Both s_c and s_r are strings of the form "u[a, b]" or "[a, b]" where
    a and b are the vector's scalars.  If the string is prefixed with
    "u", the vector is unsigned; it defines only the direction of the
    axis, not which direction is positive or negative.  Unsigned vectors
    usually exist because a test has an incomplete record.

    Returns
    -------
    basis, signed

    basis := 2×2 array.  First row is circumferential (posterior to
    anterior); second row is radial (inner to outer).

    signed := A 2-element vector.  Each element is 1 if the
    corresponding row of `basis` is signed; 0 if not.

    """
    def v_from_s(s):
        signed = {'u': False, '[': True}[s[0]]
        v = np.array(json.loads(s.lstrip("u")))
        return v, signed
    vectors = []
    signed = []
    for v, d in map(v_from_s, [s_c, s_r]):
        vectors.append(v)
        signed.append(d)
    return np.array(vectors), np.array(signed)

def test_signature(spc_id, test_id):
    """Return unique identifying string for a test."""
    if pd.isnull(test_id) or test_id == 'NA':
        test_id = 'NA'
    else:
        test_id = int(test_id)
    s = "{}_test_{}".format(spc_id, test_id)
    return s
