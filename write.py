import csv
import os

from .unit import ureg

def measurement_csv(m, f, digits=7):
    """Write measurement to file object.

    m := pint value with units and uncertainty

    f := file object or file path

    """
    def fn(f):
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC,
                               lineterminator=os.linesep)
        values = [m.value.magnitude, m.error.magnitude]
        units = str(m.units)
        num_format = "{:." + str(digits) + "g}"
        values = [num_format.format(x) for x in values]
        csvwriter.writerow(values + [units])
    if type(f) is str:
        with open(f, 'w', newline='') as  f:
            fn(f)
    else:
        fn(f)
