import csv
import re

def imtime(imlist):
    """Get timecode from list of test image filenames.

    The image filenames are assumed to be in the convention followed
    by the Elliott lab LabView camera capture VI:

    `cam#_index_timeinseconds.tiff`

    """
    if type(imlist) is str:
        imlist = [imlist]
    pattern = r"(?<=cam0_)([0-9]+)_([0-9]+\.?[0-9]+)(\.[A-Za-z]+)"
    t = []
    for s in imlist:
        timecode = re.search(pattern, s).group(2)
        t.append(float(timecode))
    return t

def imindex_lookup(csvpath, key):
    """Returns image name from image index.

    If `imagename` is not in the index, returns None.

    """
    with open(csvpath, 'r') as csvfile:
        for row in csv.reader(csvfile):
            if row[0] == key:
                return row[1]
    return None
