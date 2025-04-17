"""Test ability to read a Bluehill raw data .csv file"""

from pathlib import Path
from unittest import TestCase

from postmech import read

DIR_FIXTURES = (Path(__file__).parent / "fixtures").resolve()


def test_read_no_header():
    mdata, data = read.instron_rawdata(
        DIR_FIXTURES / "Bluehill_Universal_data_no_header.csv"
    )
    assert mdata == {}
    assert set(data.columns) == {"Time [s]", "Extension [mm]", "Load [N]"}
    assert len(data) == 51
