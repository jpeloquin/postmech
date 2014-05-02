import unittest, os
import mechana
import numpy.testing as npt

class BoseTxtTest(unittest.TestCase):
    
    fpath = os.path.join(os.path.dirname(__file__),
                         "fixtures", "bose_export.TXT")
    
    def test_bose_data(self):
        data = mechana.read.bose_data(self.fpath)
        assert len(data.columns) == 4
        npt.assert_almost_equal(data["Time"].iat[9], 0.045)
        npt.assert_almost_equal(data["Disp"].iat[-1], 5.998793 /  1000)
