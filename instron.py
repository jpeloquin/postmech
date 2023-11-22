import csv
import re

import numpy as np
import pandas as pd


def read_parameters(pth):
    """Return parameters list from an Instron .csv file

    Intended to work with both Bluehill Universal and Bluehill 3 raw data formats, even
    when a parameters block or results table is included in the header of the file.

    """
    parameters = {}
    in_list = False
    with open(pth, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for ln in reader:
            if not ln:
                continue
            m = re.match(r"^(.+) : (.+)", ln[0])
            if m is not None:
                in_list = True  # we're in a parameters list
                parameters[ln[0]] = ln[1]
            if in_list and m is None:
                in_list = False
                break
    return parameters


def read_rawdata(pth, thousands_sep=","):
    """Return raw data table from an Instron .csv file

    Intended to work with both Bluehill Universal and Bluehill 3 raw data formats, even
    when a parameters block or results table is included in the header of the file.
    Actually achieving this is a work in progres.

    Bluehill 3 column names are translated to Bluehill Universal column names:
    - "Extension" becomes "Displacement"
    - "Load" becomes "Force"

    """

    def bluehill_universal_header(ln):
        return [s.replace("(", "[").replace(")", "]") for s in ln]

    def bluehill_3_header(ln0, ln1):
        colnames = [
            s.strip().replace("Load", "Force").replace("Extension", "Displacement")
            for s in ln0
        ]
        units = [s.strip().lstrip("('").rstrip(")'") for s in ln1]
        header = [f"{nm} [{u}]" for nm, u in zip(colnames, units)]
        return header

    with open(pth, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        # Skip rows until reaching the raw data block
        for ln in reader:
            if not ln:
                continue
            if ln[0].startswith("Raw Data"):
                header = bluehill_universal_header(next(reader))
                break
            elif ln[0].startswith("Time"):
                ln1 = next(reader)
                assert ln1[0].startswith("(")
                header = bluehill_3_header(ln, ln1)
                break
            continue
        # Read the actual data
        data = {header[i]: [] for i in range(len(header))}
        for row in reader:
            if not row:
                continue
            for i, k in enumerate(header):
                v = float(_strip_sep(row[i], thousands_sep))
                data[k].append(v)
    data = pd.DataFrame.from_dict(data)
    if data.empty:
        raise Exception(
            "Empty data.  Most likely read_rawdata has a bug and failed to find the raw data table."
        )
    return data


def stretch_ratio(d, l0):
    return (d + l0) / l0


def stress(p, a):
    """Calculate 1st Piola-Kirchoff stress.

    Parameters
    ----------
    p : 1-D array or list
       Load in Newtons
    a : numeric
       Area in meters

    """
    return np.array(p) / a


def _strip_sep(s, thousands_sep=","):
    return s.replace(thousands_sep, "")
