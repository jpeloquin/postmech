"""Rename a file by prefixing its EXIF timestamp.

Run `rename_photo_timestamp.py --help` from the command line for usage.

An image named foo.jpeg will be renamed to YYYY-MM-DDTHH:MM:SS_-_foo.jpeg,
with the time taken from the image's DateTimeOriginal EXIF tag.  No time
zone conversion is performed.

"""
import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

s = "Rename a file by prefixing its EXIF timestamp."
parser = argparse.ArgumentParser(description=s)
parser.add_argument("files", nargs="+", help="File paths to rename")
args = parser.parse_args()

strfmt_iso8601 = "%Y-%m-%dT%l%M%S%z"
for p in args.files:
    im = Image.open(p)
    exif = {ExifTags.TAGS[k]: v for k, v in im.getexif().items() if k in ExifTags.TAGS}
    time_tags = ["DateTimeOriginal", "EXIF DateTimeOriginal", "DateTime"]
    for k in time_tags:
        try:
            s_time = exif[k]
            break
        except KeyError:
            continue
    else:
        raise ValueError(
            f"{p} has none of the following EXIF tags: {', '.join(time_tags)}"
        )
    # The strptime format might vary from camera to camera.  This works with a Canon
    # PowerShot ELPH 300 HS.
    t = datetime.strptime(s_time, "%Y:%m:%d %H:%M:%S")
    prefix = t.isoformat().replace(":", "")
    # ^ ISO 8601 time basic format because Windows can't cope with :
    src = Path(p)
    dest = src.parent / f"{prefix}_-_{src.name}"
    src.rename(dest)
