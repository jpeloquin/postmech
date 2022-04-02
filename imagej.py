from pathlib import Path
from typing import Union


def read_xy(pth: Union[str, Path]):
    """Read an ImageJ XY Coordinates text file"""
    pts = []
    with open(pth, "r") as f:
        for ln in f.readlines():
            x, y = (float(x) for x in ln.split())
            pts.append((x, y))
    return pts
