import pandas as pd
import numpy as np
import mechana

def bose_data(fpath):
    """Read a text data file exported by Wintest.

    Ideally, all units are converted to mks.  Currently, this is only
    implemented for displacements in mm, as time and load are mks by
    default.

    Note: Wintest terminates the first line of the file with a null
    byte, so these data files are not strictly plain text.

    """
    with open(fpath, 'rb') as f:
        lines = f.readlines()
        columns = lines[2]
        units = lines[3]
    def parseline(s):
        l = s.split(",")[:-1]
        l = [s[1:-1].strip() for s in l]
        return l
    columns = parseline(columns)
    units = parseline(units)
    data = pd.read_csv(fpath, header=2, names=columns,
                       skiprows=1, index_col=False)
    if "Disp" in columns:
        unit = units[columns.index("Disp")]
        if unit == "mm":
            data["Disp"] = data["Disp"] / 1000
    if "Elapsed Time" in columns:
        idx = columns.index("Elapsed Time")
        columns[idx] = "Time"
        data.columns = columns
    return data
