import io
import os, re, sys, csv
from os.path import join as pjoin
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image, ImageDraw

from uncertainties import ufloat

from . import instron, read
from .unit import get_ureg


def lowercase_colname(s):
    m = re.search(r"\s\(\S+\)$", s)
    if m:
        units = s[m.start() : m.end()]
        units = units.strip()[1:-1]
        return "{} ({})".format(s[0 : m.start()].lower(), units)
    else:
        return s.lower()


def decode_impath(pth):
    """Return camera number, frame number, & timestamp from image path.

    The format of the image file name is:

        cam{camera id}_{frame id}_{time}.{extension}

    The camera id and frame id are integers.  The time is a decimal
    value (include the decimal point) enumerated in seconds.  The
    extension is tiff, tif, or csv (for vic-2d exports).

    """
    s = os.path.basename(pth)
    pattern = "".join(
        [
            r"cam(?P<cam_id>[0-9]+)_",
            r"(?P<frame_id>[0-9]+)_" r"(?P<time>[0-9]+.[0-9]+)",
            r"(?:.tiff|.csv|.tif)?",
        ]
    )
    m = re.search(pattern, s)
    d = {
        "Camera ID": m.group("cam_id"),
        "Frame ID": m.group("frame_id"),
        "Time [s]": float(m.group("time")),
    }
    return d


def get_frame_size(archive):
    """Return frame size of images in image archive"""
    archive = Path(archive)
    if archive.is_dir():
        for p in archive.iterdir():
            if p.suffix in (".tiff", ".tif", ".png"):
                image = Image.open(p)
                break
        else:
            raise ValueError(
                "No .tiff, .tif, or .png files found in image archive directory."
            )
    elif archive.suffix == ".zip":
        with ZipFile(archive) as a:
            image = Image.open(io.BytesIO(a.read(a.namelist()[0])))
    else:
        raise ValueError("Image archive must be a zip file or a directory.")
    return image.size


def image_id(fpath):
    """Convert image name into a unique id."""
    s = os.path.basename(fpath)
    pattern = r"(?P<key>cam[0-9]_[0-9]+)[0-9._A-Za-z]+(?:.tiff|.csv|.tif)?"
    m = re.search(pattern, s)
    return m.group("key")


def list_images(pth):
    """Return list of image file names in a directory or zip archive"""
    pth = Path(pth)
    if pth.suffix == ".zip":
        with ZipFile(pth) as archive:
            imlist = archive.namelist()
    else:
        files = os.listdir(pth)
        pattern = r"cam[0-9]_[0-9]+_[0-9]+.[0-9]{3}.tiff"
        imlist = [f for f in files if re.match(pattern, f) is not None]
    return sorted(imlist)


def tabulate_images(pth, name_decoder=decode_impath):
    """Return table of images in a directory or zip archive"""
    imlist = list_images(pth)
    iminfo = [name_decoder(nm) for nm in imlist]
    tab = DataFrame(iminfo)
    tab["Name"] = imlist
    # Should be sorted already, but guarantee this
    tab = tab.sort_values("Time [s]")
    return tab


def tabulate_images_and_mechdata(imdir, mech_data_file=None, vic2d_dir=None):
    """Return a table with data mapped to each image frame.

    mech_data_file := Path to mechanical data table.  This can be a
    raw data file or a file with stress and strain; it just needs to
    have a 'Time [s]' column that can be matched with the image
    timestamps.  =ref_time.csv= and =image_index.csv= (both in the
    image directory) will be used to match this time to the frame
    timestamps.

    The returned table contains: camera number, frame number,
    timestamp (based on the camera), stress and strain corresponding
    to that frame (based on the ref_time.csv mediated synchronication
    with the Instron data), and path to the vic-2d data files.

    """
    if imdir is None:
        raise (Exception("Provided None as image directory."))
    if imdir.endswith(".zip"):
        p_images = pjoin(os.path.dirname(imdir), imdir[:-4])
    else:
        p_images = imdir
    p_imdata = os.path.join(p_images, "../image_measurements")
    p_imindex = pjoin(p_imdata, "image_index.csv")

    ## Load image data
    image_list = list_images(p_images)
    imindex = read_image_index(p_imindex)
    tab_frames = pd.DataFrame([a for a in map(decode_impath, image_list)])

    ## Compute frame times from the perspective of the test clock
    t_frame0 = read.measurement_csv(os.path.join(p_imdata, "ref_time.csv"))
    t_frame0 = t_frame0.nominal_value
    timestamp0 = image_time(imindex["ref_time"])

    tab_frames["Time [s]"] = (
        tab_frames["Time [s]"].astype("float") - timestamp0 + t_frame0
    )

    ## Add corresponding stress & strain values
    if mech_data_file is not None:
        mech_data = pd.read_csv(mech_data_file)
        for col in set(mech_data.columns) - set(["Time [s]"]):
            tab_frames[col] = np.interp(
                tab_frames["Time [s]"], mech_data["Time [s]"], mech_data[col]
            )

    ## Add paths to vic-2d files
    if vic2d_dir is not None:
        if vic2d_dir.endswith(".zip"):
            archive = ZipFile(vic2d_dir)
            vic2d_files = archive.namelist()
        else:
            vic2d_files = os.listdir(vic2d_dir)
        tab_v2d = pd.DataFrame([decode_impath(a) for a in vic2d_files])
        pths = [os.path.join(vic2d_dir, p) for p in vic2d_files]
        tab_v2d["Vic-2D File"] = pths
        tab_v2d = tab_v2d.drop("Time [s]", 1)
        tab_frames = pd.merge(
            tab_frames, tab_v2d, how="left", on=["Camera ID", "Frame ID"]
        )
    else:
        tab_frames["Vic-2D File"] = np.nan

    return tab_frames


def image_scale(fpath):
    """Reads `image_scale.csv` and calculates mm/px"""
    ureg = get_ureg()
    with open(fpath, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            unit = line[-1]
            if unit == "px":
                unit = ureg(unit)
                px_d = float(line[0]) * unit
                px_d = px_d.plus_minus(float(line[1]))
            else:
                unit = ureg(unit)
                d = float(line[0]) * unit
                d = d.plus_minus(float(line[1]))
    try:
        scale = d / px_d
    except UnboundLocalError:
        raise (
            Exception("{} does not have complete image scale information".format(fpath))
        )
    return scale


def from_px(fpath, scale):
    """Reads ref_length.csv

    scale := (value, sd)

    """
    d = read.measurement_csv(fpath)
    mm_d = d * scale
    return mm_d


def image_time(imname):
    """Parse image time from image filename.

    The image filenames are assumed to be in the convention followed
    by the Elliott lab LabView camera capture VI:

    `cam#_index_timeinseconds.tiff`

    """
    pattern = r"(?<=cam0_)([0-9]+)_([0-9]+\.?[0-9]+)(\.[A-Za-z]+)?"
    timecode = re.search(pattern, imname).group(2)
    return float(timecode)


def read_image_index(fpath):
    """Returns dictionary of image names for specific test events."""
    with open(fpath, "r", newline="") as csvfile:
        image_index = dict()
        for row in csv.reader(csvfile):
            image_index[row[0]] = row[1]
    return image_index


def get_image_list(fpath):
    """Returns list of images and which test phase they belong to.

    Inputs
    ------
    fpath : string
        Path for 'image_index.csv'

    Returns
    -------
    list of 2-element tuples
        A sorted list of (image_file_path, test_phase) with the test
        phase of each image defined from the `image_index.csv` file in
        the image directory.

    """
    # Read image_index.csv
    image_index = read_image_index(fpath)
    # Check for image directory existence
    imdir = os.path.dirname(fpath)
    if not os.path.isdir(imdir):
        raise Exception("Could not find image directory at " + imdir)
    # Build the image list
    images = []
    for fname in os.listdir(imdir):
        if fname.endswith((".tiff", ".tif")):
            images.append(fname)
    images.sort()
    imlist = []
    curr_phase = "extra"
    image_index = dict(
        (k, images.index(image_index[k]))
        for k, v in image_index.iteritems()
        if v != "NA"
    )

    # Find last image
    if "end" in image_index:
        i_end = image_index["end"]
    else:
        i_end = len(images) - 1

    for i, fname in enumerate(images):
        if i > i_end:
            curr_phase = "extra"
        elif i >= image_index["ref_time"] and i < image_index["ramp_start"]:
            curr_phase = "preconditioning"
        elif i >= image_index["ramp_start"]:
            curr_phase = "ramp"
        imlist.append((fname, curr_phase))
    return imlist


def make_vic2d_lists(
    p_imindex,
    d_images,
    p_mech_data,
    interval=0.01,
    highres=None,
    fout="vic2d_list.txt",
    zero_strain="zero_strain",
    start="vic2d_start",
    end="vic2d_end",
):
    """List images for vic2d analysis.

    Inputs
    ------
    highres : tuple
        If an image has a stretch ratio >= highres[0] and <=
        highres[1], it will be included in the list even if it would
        be skipped according the interval setting.
    start : string
        The key in image_index.csv naming the first frame to be
        included in the list.
    end : string
        The key in image_index.csv naming the last frame to be
        included in the list.

    """
    # Read image index
    imindex = read_image_index(p_imindex)
    if d_images.endswith(".zip"):
        archive = ZipFile(d_images)
        imlist = archive.namelist()
    else:
        imlist = [os.path.relpath(f, d_images) for f in list_images(d_images)]

    tab_frames = tabulate_images(d_images, p_mech_data)

    selected_images = [imindex[zero_strain]]

    t_start = image_time(imindex[start])
    t_end = image_time(imindex[end])
    imname_start = imindex[start]
    imname_end = imindex[end]

    y_start = tab_frames["Stretch Ratio"][imlist.index(imname_start)]
    y_end = tab_frames["Stretch Ratio"][imlist.index(imname_end)]
    yt = 0
    for i, y in enumerate(tab_frames["Stretch Ratio"]):
        t = image_time(imlist[i])
        if highres is not None:
            inhighres = y >= highres[0] and y <= highres[1]
        else:
            inhighres = False
        if t >= t_start and t <= t_end and (y - yt > interval or inhighres):
            yt = y
            selected_images.append(imlist[i])

    # Write image list
    with open(fout, "w") as f:
        for nm in selected_images:
            f.write(nm + "\n")


def mask_from_poly(poly, shape):
    """Create an image mask from a `shapely` polygon.

    poly := Polygon, MultiPolygon, or list of Polygon objects

    shape := (ny <int>, nx <int>).  Shape of the image array into which
    the polygon mask should be drawn.

    """
    if not hasattr(poly, "__iter__"):
        # Single polygon, not MultiPolygon or list of polygons.  Make it
        # a list so we can use a common execution path.
        poly = [poly]

    # Buffer the polygons to get closer to the `shapely` result; see
    # Peloquin 2017-05-07 journal entry in personal notes.
    poly = [p.buffer(0.5) for p in poly]

    # Draw the polygon(s) into the mask
    mask = Image.new("L", (shape[1], shape[0]), 0)

    def draw(mask, coords, fill=1):
        ImageDraw.Draw(mask).polygon(coords, outline=0, fill=fill)

    for p in poly:
        # Draw the entire polygon into the mask
        draw(mask, p.exterior.coords[:])
        # Anti-draw any holes into mask
        for hole in p.interiors:
            draw(mask, hole.coords[:], 0)
    return np.array(mask).astype("bool")
