import csv
import os
import subprocess

from .unit import ureg


def measurement_csv(m, f, digits=7):
    """Write measurement to file object.

    m := pint value with units and uncertainty

    f := file object or file path

    """

    def fn(f):
        csvwriter = csv.writer(
            f, quoting=csv.QUOTE_NONNUMERIC, lineterminator=os.linesep
        )
        values = [m.value.magnitude, m.error.magnitude]
        units = str(m.units)
        num_format = "{:." + str(digits) + "g}"
        values = [num_format.format(x) for x in values]
        csvwriter.writerow(values + [units])

    if type(f) is str:
        with open(f, "w", newline="") as f:
            fn(f)
    else:
        fn(f)


def try_unlock_on_fail(f, pth):
    try:
        f()
    except PermissionError:
        # File might be annexed and locked
        ret = subprocess.run(["git", "annex", "unlock", pth])
        if ret.returncode == 0:
            # Try again
            f()
        else:
            raise
