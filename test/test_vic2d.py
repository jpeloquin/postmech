# Run these tests with nose

import os
import numpy as np
from mechana.vic2d import *

fp = "/home/peloquin/elliottlab/projects/damage_and_plasticity/data/BOV_G_MEDMEN_02/vic2d_strain_subset15_decay15/cam0_031805_4773.696.csv"
# ip = "/home/peloquin/elliottlab/projects/damage_and_plasticity/data/BOV_G_MEDMEN_02/images/cam0_031805_4773.696.tiff"
fig = plot_strains(fp, fout)
