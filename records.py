# -*- coding: utf-8 -*-
import os
from os.path import join as pjoin
import logging
logger = logging.getLogger(__name__)
from zipfile import ZipFile

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
        self.test_record = {}

    @classmethod
    def from_row(cls, project_dir, row):
        self = cls(project_dir)

        ## Save test record
        self.test_record = row

        ## Primary data folder for test
        if not pd.isnull(row['ramp to failure folder']):
            self.test_dir = os.path.join(project_dir, 'data',
                                         row['ramp to failure folder'])

        ## Image list
        if not (pd.isnull(row['image directory']) or row['image directory'] == 'ND'):
            p_images = os.path.join(project_dir, 'data', row['image directory'])
            if p_images.endswith('.zip'):
                self.image_archive = p_images
                self.image_dir = p_images[:-4]
                ## We need unzipped copies of the images.  Lots of code
                ## relies on the images existing as individual files.
                need_unzip = False
                if not os.path.exists(self.image_dir):
                    need_unzip = True
                    os.mkdir(self.image_dir)
                elif len(list_images(self.image_dir)) == 0:
                    need_unzip = True
                if need_unzip:
                    logger.info("Unzipping {}".format(self.image_archive))
                    ZipFile(self.image_archive).extractall(self.image_dir)
                ## Find out where the image measurements are located,
                ## using image_index.csv as the sniff test.
                if (not pd.isnull(self.test_dir) and
                    os.path.exists(pjoin(self.test_dir, 'image_index.csv'))):
                    self.image_measurements_dir = self.test_dir
                elif (not pd.isnull(self.image_dir) and
                      os.path.exists(pjoin(self.image_dir, 'image_index.csv'))):
                    self.image_measurements_dir = self.image_dir
                else:
                    # self.image_measurements_dir = None (default)
                    pass
            else:
                self.image_dir = p_images
                self.image_measurements_dir = self.image_dir
            self.image_paths = list_images(self.image_dir)

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
        return self.test_record[key]

def test_signature(spc_id, test_id):
    """Return unique identifying string for a test."""
    if pd.isnull(test_id) or test_id == 'NA':
        test_id = 'NA'
    else:
        test_id = int(test_id)
    s = "{}_test_{}".format(spc_id, test_id)
    return s
