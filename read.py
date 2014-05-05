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
    def parseline(s):
        l = s.split(",")[:-1]
        l = [s[1:-1].strip() for s in l]
        return l

    # Read the file
    with open(fpath, 'rb') as f:
        lines = f.readlines()

    # Find header row and units
    header_row = None
    for i in xrange(5):
        line = parseline(lines[i])
        if len(line) > 0 and line[0] == "Elapsed Time":
            header_row = i
            break
    if header_row is None:
        raise Exception("No header row found in first 6 lines of "
                        + fpath)
    columns = parseline(lines[header_row])
    units = parseline(lines[header_row + 1])

    # Read data, skipping blank lines
    data = dict(zip(columns, [list() for a in columns]))
    for i in xrange(header_row + 2, len(lines)):
        line = parseline(lines[i])
        if len(line) > 0:
            for k, v in zip(columns, line):
                data[k].append(float(v))
    data = pd.DataFrame.from_dict(data)

    # Rename columns and check units
    data = data.rename(columns = {'Elapsed Time': 'Time (s)'})
    if "Load" in columns:
        assert units[columns.index("Load")] == "N"
        data = data.rename(columns = {'Load': 'Load (N)'})
    if "Disp" in columns:
        assert units[columns.index("Disp")] == "mm"
        data["Disp"] = data["Disp"] / 1000
        data = data.rename(columns = {'Disp': 'Displacement (m)'})
    return data
