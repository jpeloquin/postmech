import csv
import io
import os
import warnings
from zipfile import ZipFile

import pandas as pd
import numpy as np

from .unit import ureg


def open_archive_file(pth, mode="rt"):
    """Return a file object for a path that may include a .zip file.

    Example:

    f = archive_file('data/archive.zip/cam0_058831_3610.777.csv')

    """
    # Normalize the path
    pth = os.path.abspath(pth)

    # Get each element of the path
    parts = []
    while pth and pth != "/":
        head, tail = os.path.split(pth)
        parts.append(tail)
        pth = head

    # Walk up the path, opening zip files as necessary.  Currently, the
    # cases of 0 and 1 zip files are supported.
    pth = "/"
    while parts:
        part = parts.pop()
        pth = os.path.join(pth, part)
        if pth.endswith(".zip"):
            archive = ZipFile(pth)
            f = archive.open(os.path.join(*parts[::-1]))
            if "b" in mode:
                return io.BytesIO(f)
            elif "t" in mode:
                return io.TextIOWrapper(f)
    return open(pth, mode)


def measurement_csv(fpath):
    """Read a csv measurement file.

    The file should have the format:

    value,s.d.,"unit"

    """
    with open(fpath, "r", newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            unit = ureg(line[-1])
            d = float(line[0]) * unit
            if line[1] in set(["NA", "ND", "NaN", ""]):
                sd = 0
            else:
                sd = float(line[1])
            d = d.plus_minus(sd)
    return d


def bose_data(fpath):
    """Read a text data file exported by Wintest.

    Ideally, all units are converted to mks.  Currently, this is only
    implemented for displacements in mm, as Wintest exports time and
    load as mks by default.

    Note: Wintest may terminate the first line of the file (which has
    the as-exported filename) with a null byte, so these data files are
    not strictly plain text.

    """

    def parseline(s):
        # These files use a trailing comma
        l = [cell.strip(' "') for cell in s.rstrip("\r\n ,").split(",")]
        return l

    def is_blank(line):
        if len(line.strip()) == 0:
            return True
        else:
            return False

    dtype = {"Scan": int, "Points": int}
    # ^ everything else is float

    mks_unit = {"Sec": "s"}

    # Read the file
    with open(fpath, "r", newline="") as f:
        lines = f.readlines()

    # Find header row and units.  The header row will start with
    # "Points" if Points were exported, and (probably) "Elapsed Time"
    # otherwise.
    headerlike_text = ["Points", "Elapsed Time"]
    nsearch = 100
    for i in range(nsearch):
        line = parseline(lines[i])
        if len(line) > 0 and line[0] in headerlike_text:
            header_row = i
            break
    else:
        raise Exception(f"No header row found in first {nsearch} lines of {fpath}")
    exported_vars = parseline(lines[header_row])
    columns = ["Scan"] + exported_vars
    units = {"Scan": "1", "Points": "1"}
    units_cells = parseline(lines[header_row + 1])
    for var, unit in zip(exported_vars, units_cells):
        if var in units:
            continue
        units[var] = mks_unit.get(unit, unit)

    # Read data, skipping blank lines
    data = dict(zip(columns, [list() for a in columns]))
    i = header_row + 2
    scan = 0
    scan_istart = 0
    while i < len(lines):
        if is_blank(lines[i]):
            i += 1
            continue
        line = parseline(lines[i])
        if line[0] in headerlike_text:
            scan += 1
            scan_istart = i + 2
            i += 2
            continue
        if len(line) > 0:
            data["Scan"].append(scan)
            for k, v in zip(exported_vars, line):
                data[k].append(dtype.get(k, float)(v))
        i += 1
    data = pd.DataFrame.from_dict(data)

    # Add units to column names
    std_varname = {"Elapsed Time": "Time", "Disp": "Position"}
    newnames = {c: f"{std_varname.get(c, c)} [{units[c]}]" for c in columns}
    data = data.rename(columns=newnames)
    return data


def instron_data(fpath, thousands_sep=","):
    """Read data from an Instron csv file.

    This function is deprecated; use `instron_rawdata` instead.

    The function expects to find time, extension, and load data.  It
    assumes that time is the first column.

    Outputs
    -------
    time, extension, load : numpy array

    """
    warnings.warn(
        "instron_data is deprecated; use instron_rawdata", category=DeprecationWarning
    )

    def strip_sep(s):
        return s.replace(thousands_sep, "")

    t = []
    d = []
    p = []
    with open(fpath, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')

        def is_blank(line):
            return not (line and any(line))

        try:
            while not is_blank(reader.__next__()):
                pass
        except StopIteration:
            raise ValueError(
                "Could not find end of header (a blank line) in {}".format(fpath)
            )
        header = reader.__next__()  # read column names
        # Check that we arrived at the right row
        assert header[0] == "Time"
        # Find Load and Extension columns
        dind = header.index("Extension")
        pind = header.index("Load")
        units = reader.__next__()  # read units
        assert units[0] == "(s)"
        assert units[dind] == "(mm)"
        assert units[pind] == "(N)"
        for row in reader:
            t.append(float(strip_sep(row[0])))
            d.append(float(strip_sep(row[dind])) / 1000)  # mm -> m
            p.append(float(strip_sep(row[pind])))
    t = np.array(t)
    d = np.array(d)
    p = np.array(p)
    df = pd.DataFrame.from_dict({"Time [s]": t, "Position [m]": d, "Load [N]": p})
    return df


def instron_rawdata(fpath, thousands_sep=","):
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
        return s.replace(thousands_sep, "")

    def strip_line(line):
        """Remove trailing blank entries from line and strip leading and trailing
        whitespace from each remaining entry"""
        i = len(line)
        for e in reversed(line):
            s = e.strip()
            if s:
                return [a.strip() for a in line[:i]]
            else:
                i -= 1

    metadata = {}
    with open(fpath, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        # Read metadata rows
        try:
            # Remove trailing blank entries.  Bluehill will not generate these,
            # but if a user re-saves a Bluehill CSV file the spreadsheet software
            # will make all rows have the same number of columns, and there doesn't
            # seem to be any particular need to be picky.
            while True:
                ln = strip_line(reader.__next__())
                if not ln:
                    # Once we reach a blank line, the metadata block has ended
                    break
                k = ln[0]
                v = ln[1]
                if len(ln) > 2:
                    unit = ln[2].strip()
                    v = float(strip_sep(v)) * ureg(unit)
                metadata[k] = v
        except StopIteration:
            raise ValueError(
                "Could not find end of header (a blank line) in {}".format(fpath)
            )
        # Read column names for data table
        header = reader.__next__()
        # Check that the row we think is the header row really is the
        # header row
        assert "Time" in header
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
