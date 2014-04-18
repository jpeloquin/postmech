# Run these tests with nose

import os
import numpy as np
from mechana.vic2d import *

fp = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                  "fixtures",
                  "cam0_038228_5737.937.csv")
# fig = plot_strains(fp, "out.png")

def test_hdf5ify():
    csvfiles = [fp]
    hdf5ify(csvfiles)
