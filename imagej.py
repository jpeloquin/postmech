from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw


def read_xy(pth: Union[str, Path]):
    """Read an ImageJ XY Coordinates text file

    ImageJ coordinates are zero-indexed and have their origin in the upper left of
    the image.

    """
    pts = []
    with open(pth, "r") as f:
        for ln in f.readlines():
            x, y = (float(x) for x in ln.split())
            pts.append((x, y))
    return pts


def mask_from_xy(roi, shape):
    mask = Image.new("L", shape, 0)
    artist = ImageDraw.Draw(mask)
    artist.polygon([(x, y) for x, y in roi], fill=255, outline=None)
    return mask
