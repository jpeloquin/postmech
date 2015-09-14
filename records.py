# -*- coding: utf-8 -*-
import os

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
        self.image_paths = []
        self.vic2d_dir = None
        self.vic2d_paths = []

    @classmethod
    def from_row(cls, project_dir, row):
        self = cls(project_dir)
        ## Primary data folder for test
        self.test_dir = os.path.join(project_dir, 'data',
                                     row['ramp to failure folder'])

        ## Image list
        if not pd.isnull(row['image directory']):
            self.image_dir = os.path.join(project_dir, 'data',
                                          row['image directory'])
            self.image_paths = list_images(self.image_dir)

        ## Vic-2d directory
        if not pd.isnull(row['vic-2d export folder']):
            self.vic2d_dir = os.path.join(project_dir,
                                          row['vic-2d export folder'])

        ## Stress and strain data file
        self.stress_strain_file = os.path.join(self.test_dir,
                                               'stress_strain.csv')
        if not os.path.exists(self.stress_strain_file):
            self.stress_strain_file = None
            print("Warning: {} test {} has no stress_strain.csv "
                  "file.".format(row['specimen id'], row['test id']))

        return self

def test_signature(spc_id, test_id):
    """Return unique identifying string for a test."""
    test_id = int(test_id) if not pd.isnull(test_id) else 'NA'
    s = "{}_test_{}".format(spc_id, test_id)
    return s
