import unittest, os
import numpy.testing as npt

from postmech import read

class BoseTxtTestTimed(unittest.TestCase):

    fpath = os.path.join(os.path.dirname(__file__),
                         "fixtures", "bose_export.TXT")

    def test_bose_data(self):
        data = read.bose_data(self.fpath)
        assert len(data.columns) == 5
        # ^ Column "Scan [1]" is added by bose_data()
        npt.assert_almost_equal(data["Time [s]"].iat[9], 0.045)
        npt.assert_almost_equal(data["Position [mm]"].iat[-1],
                                5.998793)


class BoseTxtTestBlock(unittest.TestCase):

    fpath = os.path.join(os.path.dirname(__file__),
                         "fixtures", "bose_export_block.TXT")

    def test_bose_data(self):
        data = read.bose_data(self.fpath)
        assert len(data.columns) == 4
        assert sum(data['Position [mm]'].isnull()) == 0
        assert sum(data['Time [s]'].isnull()) == 0
        assert sum(data['Load [N]'].isnull()) == 0
        assert len(data['Position [mm]']) == 4600
        assert len(data['Time [s]']) == 4600
        assert len(data['Load [N]']) == 4600
        npt.assert_almost_equal(data["Time [s]"].iat[6], 0.06)
        npt.assert_almost_equal(data["Position [mm]"].iat[-1],
                                1.474395)
