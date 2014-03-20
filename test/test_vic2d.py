# Run these tests with nose

import os
import numpy as np
from mechana.vic2d import read_vic2d_export

def test_read_vic2d_export():
    fpath = os.path.join("test", "fixtures",
                         "vic2d_export.csv")
    impath = os.path.join("test", "fixtures",
                          "vic2d_export_ref.tiff")
    keys = ['x', 'exy', 'e2']
    data = read_vic2d_export(fpath, impath, keys)
    assert data['e2'][(68,329)] == -0.00202986
    assert data['e2'][(261,558)] == -0.00326161
    assert np.sum(data['exy'] != 0.0) == 92386
    assert np.sum(data['e2'] != 0.0) == 92386
