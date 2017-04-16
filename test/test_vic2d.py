# Run these tests with nose
import os
import unittest

import numpy as np
from mechana.vic2d import *

fixture_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           "fixtures")
fp = os.path.join(fixture_dir, "cam0_038228_5737.937.csv")
# fig = plot_strains(fp, "out.png")

class Vic2D2009TestCase(unittest.TestCase):
    def test_read_z2d(self):
        pth = os.path.join(fixture_dir,
                           "bov_z_latmen_01_vic2d_strain_ramp.z2d")
        z2d = read_z2d(pth)
        assert len(z2d['rois']) == 1
        self.assertEqual(z2d['rois'][0]['exterior'][0], (71, 351))
        self.assertEqual(z2d['rois'][0]['exterior'][-1], (68, 739))
