import csv

from mechana.unit import ureg

def measurement_csv(m, fpath):
    """Write measurement to csv file.

    m := (value, s.d.)

    """
    assert m[0].units == m[1].units
    with open(fpath, 'w') as  f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow([m[0].magnitude,
                            m[1].magnitude,
                            str(m[0].units)])
