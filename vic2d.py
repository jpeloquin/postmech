from io import BytesIO
import re, os
import math
from os import path
from zipfile import ZipFile
import hashlib
import warnings

import numpy as np
import numpy.ma as ma
import shapely
from numpy.linalg import inv
import pandas as pd
from scipy.ndimage import map_coordinates
import matplotlib as mpl
import matplotlib.pyplot as plt
from lxml import etree as ET
from shapely.geometry import Point

from lbplt.colormaps import choose_cmap, cmap_div


def listcsvs(directory):
    """List csv files in a directory."""
    files = sorted(os.listdir(directory))
    pattern = r"cam[0-9]_[0-9]+_[0-9]+.[0-9]{3}.csv"
    files = [f for f in files if re.match(pattern, f) is not None]
    files = [os.path.join(directory, f) for f in files]
    return sorted(list(files))


def read_csv(f):
    """Return list of data tables from a Vic-2D csv file.

    f := str or file-like buffer.  A str is treated as a file path.  A
    bytes object is treated as raw data.

    This function supports multi-ROI csv files.  Because Vic-2D
    implements multi-ROI csv files in an inconvenient way, read_table
    has to read the file twice and is slower than it should be.

    """
    if isinstance(f, bytes):
        name = "bytes array"
        s = f
    elif isinstance(f, str):
        name = f
        with open(name, "rb") as f:
            s = f.read()
    else:
        # Primarily meant for zipfile.ZipExtFile. For example,
        # with ZipFile("filename.zip") as f:
        #    read_csv(f.open("filename_in_zip")
        name = f.name
        s = f.read()
    # To handle multi-ROI csv files, split bytes on '\n\n'.  The file is
    # always terminated with '\n\n', so the last item in the split is
    # always blank.  We work in bytes and defer to read_table and pandas
    # to handle encoding.
    sections = s.split(b"\n\n")[:-1]
    if len(sections) == 0:
        warnings.warn("{} has zero rows of data.".format(name))
    tables = [read_table(BytesIO(x)) for x in sections]
    return tables


def read_table(f):
    """Return data table from a Vic-2D csv file with 1 ROI."""
    df = pd.read_csv(
        f,
        skipinitialspace=True,
        dtype={"x": int, "y": int},
        skip_blank_lines=True,
    )
    # ^ vic2d adds an extra line at the end, which gets read as a row
    # of missing values.  Hence skip_blank_lines.
    return df


def _roi(roi):
    """Return ROI data from XML AOI element (Vic-2D 2009)."""
    # Initialize ROI dictionary
    d = {
        "type": "polygon",
        "subset size": None,
        "spacing": None,
        "exterior": [],
        "interior": [],
    }
    d["type"] = roi.get("type")
    d["subset size"] = int(roi.get("subsetsize"))
    d["spacing"] = int(roi.get("spacing"))
    # Exterior boundary
    l = roi.find("points").text.split(" ")
    d["exterior"] = [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
    # Interior boundary (cut-outs)
    # TODO: handle multiple cut-outs
    e_int = roi.find("cut")
    if e_int is not None:
        l = e_int.text.split(" ")
        d["interior"] = [(int(l[i]), int(l[i + 1])) for i in range(0, len(l), 2)]
    return d


def read_z2d(pth):
    """Read the ROI from a z2d file."""
    data = {}
    with ZipFile(pth, "r").open("project.xml") as f:
        root = ET.parse(f)
        # Get ROIs
        # (a) Get ROI from <aoi> tag (Vic-2D 2009)
        data["rois"] = [_roi(x) for x in root.findall("projectaois/aoi")]
        # (b) Get ROI from <aoinode> tag (Vic-2D 6)
        for aoi_tag in root.findall("projectaois/aoinode"):
            # Does not handle interior cuts or non-polygonal AOIs yet
            roi = {
                "type": "polygon",
                "subset size": int(
                    aoi_tag.find("polygonmask/polygon").get("subsetsize")
                ),
                "spacing": int(aoi_tag.find("polygonmask/polygon").get("stepsize")),
                "exterior": [],
                "interior": [],
            }
            pts = [
                int(p) for p in aoi_tag.find("polygonmask/polygon/outline").text.split()
            ]
            roi["exterior"] = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
            data["rois"].append(roi)
        # Get image list
        data["images"] = [
            x.text
            for x in root.find("files").getchildren()
            if x.tag in set(["reference", "deformed"])
        ]
    return data


def hashfile(fpath):
    with open(fpath, "rb") as f:
        fhash = hashlib.sha1(f.read()).hexdigest()
    return fhash


def label_regions_strain_tab(tab, polys, inplace=False):
    """Assign pixels in a strain table to polygonal regions.

    regions := Dictionary.  Key := region labels.  Value := Polygon or MultiPolygon object.

    This function is fairly slow; for large tables, it is recommended to
    use arrays and masks.

    """
    if not inplace:
        tab = tab.copy()
    tab["region"] = ""
    for region in polys:
        poly = polys[region]
        poly = shapely.prepared.prep(poly)
        bb = polys[region].bounds
        m = np.logical_and.reduce(
            [tab["x"] > bb[0], tab["x"] < bb[2], tab["y"] > bb[1], tab["y"] < bb[3]]
        )
        idx = [
            i for i, r in tab[m].iterrows() if poly.contains(Point((r["x"], r["y"])))
        ]
        tab.loc[idx, "region"] = region
    return tab


### Deprecated.  Still used in filter size sensitivity and subset size
### sensitivity analysis.
def summarize_vic2d(vicdir, imdir):
    """Calculate summary statistics for Vic-2D data.

    Note: If the Vic-2D data were exported including blank regions,
    you will find many, many zeros in the data.

    """
    pth = path.join(imdir, "..", "stress_strain.csv")
    tab_mech = pd.read_csv(pth)
    imstrain = dict(frame_stats(imdir, tab_mech))
    fields = ["exx", "eyy", "exy"]
    # Initialize output
    q05 = {k: [] for k in fields}
    q95 = {k: [] for k in fields}
    q50 = {k: [] for k in fields}
    strain = []
    keys = []
    vdset = Vic2DDataset(vicdir)
    for k in vdset.keys:
        df = vdset[k]
        keys.append(k)
        strain.append(imstrain[k])
        for field in fields:
            q = np.percentile(df[field], [5, 50, 95])
            q05[field].append(q[0])
            q50[field].append(q[1])
            q95[field].append(q[2])
    out = {"key": keys, "strain": strain, "q05": q05, "median": q50, "q95": q95}
    return out


def summarize_strain_field(data):
    """Return strain field summary statistics."""
    out = {}
    cols_out = ["exx", "eyy", "|exy|"]
    data["|exy|"] = np.abs(data["exy"])
    # Summary statistic functions
    fn_from_key = {
        "median": lambda x: x.median(),
        "mean": lambda x: x.mean(),
        "sd": lambda x: x.std(),
        "0.025 quantile": lambda x: x.quantile(0.025),
        "0.975 quantile": lambda x: x.quantile(0.975),
        "0.16 quantile": lambda x: x.quantile(0.16),
        "0.84 quantile": lambda x: x.quantile(0.84),
        "0.25 quantile": lambda x: x.quantile(0.25),
        "0.75 quantile": lambda x: x.quantile(0.75),
    }
    # Compute summary statistics for each strain component
    row = {"n": np.sum(~pd.isnull(data["exx"]))}
    # ^ Assumption: Each tracked pixel always has all three strain
    # components defined.
    for c in cols_out:
        for k, fn in fn_from_key.items():
            row["{}, {}".format(c, k)] = fn(data[c])
    return pd.DataFrame([row])


def clip_bbox_to_int(bbox):
    """Convert bounding box to integer values.

    Discard any partial-pixel edges.

    """
    # Use integers as bounding box.  Use only pixels completely inside
    # the bounding box.
    bbox[0] = int(math.ceil(bbox[0]))
    bbox[1] = int(math.floor(bbox[1]))
    bbox[2] = int(math.ceil(bbox[2]))
    bbox[3] = int(math.floor(bbox[3]))
    return bbox


def strainimg(df, field, extent=None):
    """Create a strain image from a list of values.

    field := Column name in `df` containing the strain data to plot.

    extent := Bounding box [xmin, xmax, ymin, ymax].  Values are
    inclusive.  Only pixels with coordinates on the boundary or in the
    interior of the bounding box are used for the strain image.  Hence,
    a bounding box of [-10.5, 10.5, 10.5, 20.5] is equivalent to [-10,
    10, 11, 20].

    Note that `extent` is used the same as in matplotlib's `imshow`
    function.

    """
    if extent is None:
        extent = [min(df["x"]), max(df["x"]), min(df["y"]), max(df["y"])]

    bbox = clip_bbox_to_int(extent)

    # Vic-2D indexes x and y from 0
    strainfield = np.empty((bbox[3] - bbox[2] + 1, bbox[1] - bbox[0] + 1))
    strainfield.fill(np.nan)
    x = df["x"] - bbox[0]
    y = df["y"] - bbox[2]
    v = df[field]
    strainfield[[y, x]] = v
    extent = bbox
    return strainfield, extent


def transform_image(img, basis, order=3):
    """Transform image so basis[0] is right and basis[1] is up.

    img := Masked array; the image to be transformed.

    basis := 2x2 array mapping Vic-2D image xy coordinates to the
    plot axes.  That is, the row vectors of `basis` define the
    plot's x and y vectors in the image's xy coordinate system U,
    where U is defined by x = right and y = down in the as-displayed
    image.

    """
    # prefixes:
    # i := original image ij (units = px); i → y, j → x
    # x := original image xy (units = px); aka U
    # j := transformed image ij; i → x, j → y
    # y := transformed image xy; aka V

    # Use a masked array for the strain field so the spline-based image
    # transforms work.  The transformation undoes the mask.
    mask = np.isnan(img)
    img[mask] = 0
    img = ma.array(img, mask=mask)

    i_shp = img.shape
    i_bb = np.array([[i_shp[0], i_shp[1]], [0, i_shp[1]], [0, 0], [i_shp[0], 0]]).T
    x_aff_i = np.array([[0, 1], [1, 0]])
    x_bb = np.dot(x_aff_i, i_bb)
    y_aff_x = basis
    y_bb = np.dot(y_aff_x, x_bb)
    y_max = np.ceil(np.max(y_bb, axis=1)).astype("int")
    y_min = np.floor(np.min(y_bb, axis=1)).astype("int")
    y_grid = np.mgrid[y_min[0] : y_max[0], y_min[1] : y_max[1]]
    j_shp = y_grid.shape[1:]

    i_transf = inv(x_aff_i) @ inv(y_aff_x) @ y_grid.reshape(2, -1)
    img_transf = map_coordinates(img, i_transf, cval=np.nan, order=order).reshape(j_shp)
    mask_transf = map_coordinates(img.mask, i_transf, cval=np.nan, order=0).reshape(
        j_shp
    )
    img_transf[mask_transf] = np.nan
    # Return to standard image convention
    return img_transf.swapaxes(0, 1)


def img(df, col, shp):
    """Convert a column in a Vic-2D csv export to an image."""
    # Note: Vic-2D indexes x and y from 0
    i = np.empty(shp)
    i.fill(np.nan)
    i[(df["y"], df["x"])] = df[col]
    return i


def read_strain_components(pth):
    """Read all strain component images from a Vic-2D csv file."""
    table = pd.concat(read_csv(pth))
    if len(table) != 0:
        bbox = [
            np.min(table["x"].values),
            np.max(table["x"].values),
            np.min(table["y"].values),
            np.max(table["y"].values),
        ]
        exx, ext = strainimg(table, "exx", bbox)
        eyy, ext = strainimg(table, "eyy", bbox)
        exy, ext = strainimg(table, "exy", bbox)
    else:
        exx = []
        eyy = []
        exy = []
    components = {"exx": exx, "eyy": eyy, "exy": exy}
    return components


def plot_strains(csvpath):
    """Return three-panel strain fields figure from a Vic-2D .csv file."""
    df = pd.concat(read_csv(csvpath))

    # Find extent of region that has values
    xmin = min(df["x"])
    xmax = max(df["x"])
    ymin = min(df["y"])
    ymax = max(df["y"])

    ## Initialize figure
    fig = plt.figure(figsize=(6.0, 2.5), dpi=300, facecolor="w")
    ax1 = fig.add_subplot(131, aspect="equal")
    ax2 = fig.add_subplot(132, aspect="equal")
    ax3 = fig.add_subplot(133, aspect="equal")
    axes = [ax1, ax2, ax3]
    mpl.rcParams.update({"font.size": 10})

    ## Add the three strain plots
    fields = ["exx", "eyy", "exy"]
    ctitles = ["$e_{xx}$", "$e_{yy}$", "$e_{xy}$"]
    for i, field in enumerate(fields):
        ## Plot strain image
        im, extent = strainimg(df, field)
        ax = axes[i]
        cmin, cmax = np.percentile(df[field].values, [5, 95])
        extremum = max(abs(cmin), abs(cmax))
        leftextend = np.nanmin(im) < -extremum
        rightextend = np.nanmax(im) > extremum
        if leftextend and not rightextend:
            extend = "min"
        elif rightextend and not leftextend:
            extend = "max"
        elif leftextend and rightextend:
            extend = "both"
        else:
            extend = "neither"

        implot = ax.imshow(im, cmap=cmap_div, vmin=-extremum, vmax=extremum)

        ## Format axis
        ax.axis("off")
        ax.axis((xmin, xmax, ymin, ymax))
        ax.invert_yaxis()

        ## Add colorbar
        ticker = mpl.ticker.MaxNLocator(nbins=4)
        cbar = fig.colorbar(
            implot, orientation="horizontal", extend=extend, ticks=ticker, ax=ax
        )
        cbar.set_label(ctitles[i], size=14)

        ## Set colorbar limits
        clim = max(abs(cmin), abs(cmax))
        cbar.set_clim((-clim, clim))

    ## Format figure
    plt.tight_layout()

    return fig


def plot_vic2d_data(
    simg,
    component,
    gimg=None,
    scale=None,
    fig_width=5,
    fig_height=4,
    fig_fontsize=12,
    cmap=None,
    norm=None,
    basis=np.eye(2),
    extent=None,
):
    """Plot a strain field from a Vic-2D data table.

    basis := 2×2 array.  First row is circumferential (posterior to
    anterior); second row is radial (inner to outer); both in image xy
    coordinates.

    """
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = fig.add_subplot(111)
    ax.axis("off")

    limits = (np.nanpercentile(simg, 5), np.nanpercentile(simg, 95))
    if cmap is None and norm is None:
        cmap, norm = choose_cmap(limits)

    # Transform the strain field so it is aligned with provided
    # axes.
    simg = transform_image(simg, basis)
    # Similarly, transform the extent
    if extent is not None:
        bbox = (
            np.array(
                [
                    [extent[1], extent[3]],
                    [extent[0], extent[3]],
                    [extent[0], extent[2]],
                    [extent[1], extent[2]],
                ]
            )
            @ basis.T
        )
        minima = np.min(bbox, axis=0)
        maxima = np.max(bbox, axis=0)
        extent = [minima[0], maxima[0], minima[1], maxima[1]]

    # Plot photo
    if gimg is not None:
        gimg = transform_image(gimg, basis)
        aximg_gray = ax.imshow(gimg, cmap="gray", origin="lower")

    # Plot strain field
    aximg_strain = ax.imshow(simg, cmap=cmap, norm=norm, origin="lower", extent=extent)

    ## Add 5 mm scale bar
    if scale is not None:
        try:
            px_barw = (5 * scale._REGISTRY("mm") * scale).to_base_units()
            assert px_barw.units == "pixel"
        except AssertionError:
            px_barw = (5 * scale._REGISTRY("mm") / scale).to_base_units()
        assert px_barw.units == "pixel"
        px_barw = px_barw.m
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, AnchoredOffsetbox

        transform = ax.transData
        bars = AuxTransformBox(transform)
        ylim = ax.get_ylim()
        px_yrange = ylim[1] - ylim[0]
        px_barh = px_yrange * 0.03
        ax.set_ylim(ylim[0] - px_barh * 2.5, ylim[1])
        bars.add_artist(Rectangle((0, 0), px_barw, px_barh, fc="black"))
        offsetbox = AnchoredOffsetbox(
            4, pad=0.1, borderpad=0.1, child=bars, frameon=False
        )
        ax.add_artist(offsetbox)

    ## Add colorbar
    if not np.all(simg[np.logical_not(np.isnan(simg))] == 0):
        ticker = mpl.ticker.MaxNLocator(nbins=5)
    else:
        ticker = mpl.ticker.FixedLocator([-1, 0, 1])
    cb = fig.colorbar(
        aximg_strain,
        ticks=ticker,
        extend="both",
        label=r"$E_{" + component[1:] + "}$",
        orientation="horizontal",
    )
    font = mpl.font_manager.FontProperties(size=fig_fontsize * 1.8)
    cb.ax.yaxis.label.set_font_properties(font)
    cb.ax.xaxis.label.set_font_properties(font)
    fig.tight_layout()
    return fig


def setup_vic2d(pth, imlist, imarchive, z2d_template=None):
    """Write a Vic-2D image list to a z2d file with the actual images.

    imarchive := ZipFile object of image archive.

    z2d_template := path to a .z2d file to use as a template.  The ROI
    and seed point location defined in the template will be preserved.

    """
    # Make sure the output directory exists
    d, f = os.path.split(pth)
    fname, ext = os.path.splitext(f)
    dir_images = os.path.join(d, f"{fname}_images")
    # ^ images in imlist will be copied here
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)
    if z2d_template is not None:
        # Write the z2d file to the output directory
        with ZipFile(z2d_template, "r").open("project.xml") as f:
            template = f.read()
        xml = replace_imlist_z2dxml(
            template, [os.path.join(f"{fname}_images", i) for i in imlist]
        )
        with ZipFile(os.path.join(d, fname + ".z2d"), "w") as f:
            f.writestr("project.xml", xml)
    # Write the image list to the output directory
    with open(os.path.join(d, f"{fname}_images.txt"), "w") as f:
        for ln in imlist:
            f.write(fname + "/" + ln + "\n")
    # Write the images to the output directory
    for nm in imlist:
        with open(os.path.join(dir_images, nm), "wb") as f:
            f.write(imarchive.read(nm))


def tracked_mask(tab, size):
    """Return boolean image mask of tracked pixels

    tab := data table of a Vic-2D export file

    """
    mask = np.full((size[1], size[0]), False)
    mask[(tab["y"], tab["x"])] = True
    return mask


def count_tracked(pth, size):
    """Return image mask where each pixel = # frames pixel was tracked.

    pth := path of the zipped Vic-2D data export

    size := (width, height) of images analyzed by Vic-2D

    """
    count = np.zeros((size[1], size[0]), dtype="int")
    with ZipFile(pth) as f:
        for nm in f.namelist():
            tab = pd.concat(read_csv(f.open(nm)))
            count = count + tracked_mask(tab, size).astype("int")
    return count


def replace_imlist_z2dxml(xml, imlist):
    """Replace the <files> tag in Vic-2D XML with a new image list.

    The first image in `imlist` will be used as the reference image.
    Only the <files> tag and its children will be modified.

    """
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(BytesIO(xml), parser)  # so pretty-printing works
    root = tree.getroot()
    # Remove the existing <files> element
    e = root.find("files")
    root.remove(e)
    # Build the new image list
    e_files = ET.Element("files", attrib={"lri": "files"})
    ET.SubElement(e_files, "reference").text = imlist[0]
    for i in imlist:
        ET.SubElement(e_files, "deformed").text = i
    # Insert the new image list
    root.insert(0, e_files)
    return ET.tostring(tree, pretty_print=True, doctype="<!DOCTYPE vpml>")
