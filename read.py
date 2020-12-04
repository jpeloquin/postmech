import csv
import io
import os
import warnings
from zipfile import ZipFile

import pandas as pd
import numpy as np
from pint import UnitRegistry

import mechana
from mechana.unit import ureg

def open_archive_file(pth, mode='rt'):
    """Return a file object for a path that may include a .zip file.

    Example:

    f = archive_file('data/archive.zip/cam0_058831_3610.777.csv')

    """
    # Normalize the path
    pth = os.path.abspath(pth)

    # Get each element of the path
    parts = []
    while pth and pth != '/':
        head, tail = os.path.split(pth)
        parts.append(tail)
        pth = head

    # Walk up the path, opening zip files as necessary.  Currently, the
    # cases of 0 and 1 zip files are supported.
    pth = '/'
    while parts:
        part = parts.pop()
        pth = os.path.join(pth, part)
        if pth.endswith('.zip'):
            archive = ZipFile(pth)
            f = archive.open(os.path.join(*parts[::-1]))
            if 'b' in mode:
                return io.BytesIO(f)
            elif 't' in mode:
                return io.TextIOWrapper(f)
    return open(pth, mode)

def measurement_csv(fpath):
    """Read a csv measurement file.

    The file should have the format:

    value,s.d.,"unit"

    """
    with open(fpath, 'r', newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            unit = ureg(line[-1])
            d = float(line[0]) * unit
            if line[1] in set(['NA', 'ND', 'NaN', '']):
                sd = 0
            else:
                sd = float(line[1])
            d = d.plus_minus(sd)
    return d

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
    with open(fpath, 'r', newline='') as f:
        lines = f.readlines()

    # Find header row and units
    header_row = None
    for i in range(5):
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
    for i in range(header_row + 2, len(lines)):
        line = parseline(lines[i])
        if len(line) > 0:
            for k, v in zip(columns, line):
                data[k].append(float(v))
    data = pd.DataFrame.from_dict(data)

    # Rename columns and check units
    data = data.rename(columns = {'Elapsed Time': 'Time [s]'})
    if "Load" in columns:
        assert units[columns.index("Load")] == "N"
        data = data.rename(columns = {'Load': 'Load [N]'})
    if "Disp" in columns:
        assert units[columns.index("Disp")] == "mm"
        data["Disp"] = data["Disp"] / 1000
        data = data.rename(columns = {'Disp': 'Position [m]'})
    return data


def instron_data(fpath, thousands_sep=','):
    """Read data from an Instron csv file.

    This function is deprecated; use `instron_rawdata` instead.

    The function expects to find time, extension, and load data.  It
    assumes that time is the first column.

    Outputs
    -------
    time, extension, load : numpy array

    """
    warnings.warn("instron_data is deprecated; use instron_rawdata",
                  category=DeprecationWarning)

    def strip_sep(s):
        return s.replace(thousands_sep, '')

    t = []
    d = []
    p = []
    with open(fpath, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        def is_blank(line):
            return not (line and any(line))
        try:
            while not is_blank(reader.__next__()):
                pass
        except StopIteration:
            raise ValueError("Could not find end of header (a blank line) in {}".format(fpath))
        header = reader.__next__() # read column names
        # Check that we arrived at the right row
        assert header[0] == 'Time'
        # Find Load and Extension columns
        dind = header.index('Extension')
        pind = header.index('Load')
        units = reader.__next__() # read units
        assert units[0] == "[s]"
        assert units[dind] == "[mm]"
        assert units[pind] == "[N]"
        for row in reader:
            t.append(float(strip_sep(row[0])))
            d.append(float(strip_sep(row[dind])) / 1000) # mm -> m
            p.append(float(strip_sep(row[pind])))
    t = np.array(t)
    d = np.array(d)
    p = np.array(p)
    df = pd.DataFrame.from_dict({'Time [s]': t,
                                 'Position [m]': d,
                                 'Load [N]': p})
    return df


def instron_rawdata(fpath, thousands_sep=','):
    """Read a Instron / Bluehill raw data csv file.

    Outputs
    -------
    metadata, data

    metadata := a dictionary of metadata keys and values.  For example,
    metadata["General : Start time"] would return something like "Wed,
    October 24, 2018 17:38:51".

    data := a pandas table with the test data.  The columns are named in
    the pattern "Channel (unit)".

    """
    def strip_sep(s):
        return s.replace(thousands_sep, '')
    def is_blank(line):
        return not (line and any(line))
    metadata = {}
    with open(fpath, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        # Read metadata rows
        try:
            # Once we reach a blank line, the metadata block has ended
            ln = reader.__next__()
            while not is_blank(ln):
                k = ln[0].strip()
                v = ln[1].strip()
                if len(ln) > 2:
                    unit = ln[2].strip()
                    v = float(v) * ureg(unit)
                metadata[k] = v
                ln = reader.__next__()
        except StopIteration:
            raise ValueError("Could not find end of header (a blank line) in {}".format(fpath))
        # Read column names for data table
        header = reader.__next__()
        # Check that the row we think is the header row really is the
        # header row
        assert 'Time' in header
        # Read units header
        units = [s.strip().lstrip("('").rstrip(")'") for s in reader.__next__()]
        for i, s in enumerate(header):
            header[i] = f"{s.strip()} [{units[i]}]"
        # Read the actual data
        data = {header[i]: [] for i in range(len(header))}
        for row in reader:
            for i, k in enumerate(header):
                v = float(strip_sep(row[i]))
                data[k].append(v)
    data = pd.DataFrame.from_dict(data)
    return metadata, data
