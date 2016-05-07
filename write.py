import csv
import os

from .unit import ureg

def measurement_csv(m, fpath, digits=7):
    """Write measurement to csv file.

    m := pint value with units and uncertainty

    """
    with open(fpath, 'w', newline='') as  f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,
                               lineterminator=os.linesep)
        values = [m.value.magnitude, m.error.magnitude]
        units = str(m.units)
        num_format = "{:." + str(digits) + "g}"
        values = [num_format.format(x) for x in values]
        csvwriter.writerow(values + [units])
