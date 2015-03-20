import csv

from .unit import ureg

def measurement_csv(m, fpath):
    """Write measurement to csv file.

    m := pint value with units and uncertainty

    """
    with open(fpath, 'w') as  f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow([m.value.magnitude,
                            m.error.magnitude,
                            str(m.units)])
