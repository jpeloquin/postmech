import csv
from io import BytesIO, StringIO
import re, os
import math
from os import path
from zipfile import ZipFile
from collections import defaultdict
import hashlib
import warnings

import numpy as np
import numpy.ma as ma
from numpy.linalg import inv
import pandas as pd
from scipy.ndimage.interpolation import map_coordinates
from scipy.sparse import coo_matrix
from PIL import Image
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lxml import etree as ET

import mechana
from mechana.unit import ureg
from mechana.images import image_id
from lbplt.colormaps import choose_cmap, cdict_div

mpl.cm.register_cmap(name="lab_diverging",
                     data=cdict_div, lut=256)

def listcsvs(directory):
    """List csv files in a directory.

    """
    files = sorted(os.listdir(directory))
    pattern = r'cam[0-9]_[0-9]+_[0-9]+.[0-9]{3}.csv'
    files = [f for f in files
             if re.match(pattern, f) is not None]
    files = [os.path.join(directory, f) for f in files]
    return sorted(list(files))

def read_csv(f):
    """Return list of data tables from a Vic-2D csv file.

    f := str or file-like buffer.  A str is treated as a file path.

    This function supports multi-ROI csv files.  Because Vic-2D
    implements multi-ROI csv files in an inconvenient way, read_table
    has to read the file twice and is slower than it should be.

    """
    if type(f) is str:
        pth = f
        with open(pth) as f:
            s = f.read()
    else:
        pth = f.name
        s = f.read()
    # To handle multi-ROI csv files, split string on '\n\n'.  The file
    # is always terminated with '\n\n', so the last item in the split is
    # always blank.
    sections = s.split('\n\n')[:-1]
    if len(sections) == 0:
        warnings.warn("{} has zero rows of data.".format(pth))
    tables = [read_table(StringIO(x)) for x in sections]
    return tables

def read_table(f):
    """Return data table from a Vic-2D csv file with 1 ROI."""
    df = pd.read_csv(f, skipinitialspace=True).dropna(how='all')
    # ^ vic2d adds an extra line at the end, which gets read as a row
    # of missing values.  Hence the dropna call.
    df['x'] = df['x'].astype(np.int)
    df['y'] = df['y'].astype(np.int)
    return df

def _roi(roi):
    """Return ROI data from xml aoi element."""
    d = {}
    d['type'] = roi.get('type')
    d['subset size'] = int(roi.get('subsetsize'))
    d['spacing'] = int(roi.get('spacing'))
    # Exterior boundary
    l = roi.find('points').text.split(" ")
    d['exterior'] = [(int(l[i]), int(l[i+1]))
                     for i in range(0, len(l), 2)]
    # Interior boundary (cut-outs)
    # TODO: handle multiple cut-outs
    e_int = roi.find('cut')
    if e_int is not None:
        l = e_int.text.split(" ")
        d['interior'] = [(int(l[i]), int(l[i+1]))
                         for i in range(0, len(l), 2)]
    else:
        d['interior'] = None
    return d

def read_z2d(pth):
    """Read the ROI from a z2d file."""
    data = {}
    with ZipFile(pth, 'r').open('project.xml') as f:
        root = ET.parse(f)
        # Get ROIs
        data['rois'] = [_roi(x) for x in root.findall('projectaois/aoi')]
        # Get image list
        data['images'] = [x.text for x in root.find('files').getchildren()
                          if x.tag in set(['reference', 'deformed'])]
    return data

class Vic2DDataset:
    """A class to represent a set of Vic-2D data files.

    """
    h5store = None
    keys = []
    hashes = []
    csvpaths = []

    def __init__(self, vicdir):
        # The class uses an hdf5 file as the data storage location
        h5path = os.path.join(vicdir, 'data.h5')
        if not os.path.exists(h5path):
            print(h5path + ' does not exist; building.')
            hdf5ify(vicdir)
            self.h5store = h5py.File(h5path, 'r')
        else:
            self.load_h5(h5path)
            # Make sure the h5 file is up to date
            uptodate = True
            csvfiles = [f for f in listcsvs(vicdir)]
            csvhashes = [hashfile(f) for f in csvfiles]
            keys = [mechana.images.image_id(f) for f in csvfiles]
            h5hashdict = dict(zip(self.keys, self.hashes))
            csvhashdict = dict(zip(keys, csvhashes))
            allkeys = set(keys + self.keys.value.tolist())
            for k in allkeys:
                if k not in self.keys or k not in keys:
                    # a frame exists in only one location
                    uptodate = False
                if h5hashdict[k] != csvhashdict[k]:
                    # data is out of sync
                    uptodate = False
            if not uptodate:
                print(h5path + ' not up to date; rebuilding.')
                self.h5store.close()
                os.remove(h5path)
                hdf5ify(vicdir)
                self.load_h5(h5path)

    def load_h5(self, h5path):
        self.h5store = h5py.File(h5path, 'r')
        self.keys = self.h5store["keys"]
        self.hashes = self.h5store["csvhashes"]
        self.csvpaths = self.h5store["csvpaths"]

    def __getitem__(self, key):
        group = self.h5store[key]
        fields = group.keys()
        columns = [group[k].value for k in fields]
        df = pd.DataFrame.from_dict(dict(zip(fields, columns)))
        return df

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        """Close the h5 file on object destruction.

        """
        self.h5store.close()

def hashfile(fpath):
    with open(fpath, 'rb') as f:
        fhash = hashlib.sha1(f.read()).hexdigest()
    return fhash

def hdf5ify(fdir, h5path=None):
    """Read csv files and store them in an hdf5 file.

    Storing the data in a binary format is intended to permit faster
    reads and interoperability between sofware packages.

    """
    savecolumns = ['x', 'y', 'u', 'v', 'exx', 'eyy', 'exy']

    # Define output path
    if h5path is None:
        h5path = os.path.join(fdir, 'data.h5')
    # Create list of vic2d csv files in the directory
    csvfiles = [f for f in listcsvs(fdir)]
    key = [image_id(f) for f in csvfiles]
    fhashes = [hashfile(f) for f in csvfiles]
    hashdict = dict(zip(key, fhashes))
    fpdict = dict(zip(key, csvfiles))
    mdata = pd.DataFrame.from_dict({'key': key,
                                    'hash': fhashes,
                                    'csvpath': csvfiles})
    with h5py.File(h5path, 'w') as h5store:
        h5store.create_dataset("keys", data=key)
        h5store.create_dataset("csvpaths",
            data=[os.path.basename(f) for f in csvfiles])
        h5store.create_dataset("csvhashes", data=fhashes)
        for k, fp in zip(key, csvfiles):
            df = readv2dcsv(fp)
            grp = h5store.create_group(k)
            for c in savecolumns:
                try:
                    grp.create_dataset(c, data=df[c])
                except KeyError:
                    print('Error reading file: ' + fp)
                    raise

def label_regions_strain_tab(tab, polys, inplace=False):
    """Assign pixels in a strain table to polygonal regions.

    regions := Dictionary.  Key := region labels.  Value := Polygon or MultiPolygon object.

    This function is fairly slow; for large tables, it is recommended to
    use arrays and masks.

    """
    if not inplace:
        tab = tab.copy()
    tab['region'] = ''
    for region in polys:
        poly = polys[region]
        poly = shapely.prepared.prep(poly)
        bb = polys[region].bounds
        m = np.logical_and.reduce([tab_strain['x'] > bb[0],
                                   tab_strain['x'] < bb[2],
                                   tab_strain['y'] > bb[1],
                                   tab_strain['y'] < bb[3]])
        idx = [i for i, r in tab_strain[m].iterrows()
               if poly.contains(Point((r['x'], r['y'])))]
        tab.loc[idx, 'region'] = region
    return tab

### Deprecated.  Still used in filter size sensitivity and subset size
### sensitivity analysis.
def summarize_vic2d(vicdir, imdir):
    """Calculate summary statistics for Vic-2D data.

    Note: If the Vic-2D data were exported including blank regions,
    you will find many, many zeros in the data.

    """
    pth = path.join(imdir, '..', 'stress_strain.csv')
    tab_mech = pd.read_csv(pth)
    imstrain = dict(frame_stats(imdir, tab_mech))
    fields = ['exx', 'eyy', 'exy']
    # Initialize output
    q05 = {k: [] for k in fields}
    q95 = {k: [] for k in fields}
    q50 = {k: [] for k in fields}
    strain = []
    keys = []
    vdset = mechana.vic2d.Vic2DDataset(vicdir)
    for k in vdset.keys:
        df = vdset[k]
        keys.append(k)
        strain.append(imstrain[k])
        for field in fields:
            q = np.percentile(df[field], [5, 50, 95])
            q05[field].append(q[0])
            q50[field].append(q[1])
            q95[field].append(q[2])
    out = {'key': keys,
           'strain': strain,
           'q05': q05,
           'median': q50,
           'q95': q95}
    return out

def summarize_strain_field(data):
    """Return strain field summary statistics.

    """
    out = {}
    cols_out = ['exx', 'eyy', '|exy|']
    data['|exy|'] = np.abs(data['exy'])
    # Summary statistic functions
    fn_from_key = {'n': lambda x: x.count(),
                   'median': lambda x: x.median(),
                   'mean': lambda x: x.mean(),
                   'sd': lambda x: x.std(),
                   '0.025 quantile': lambda x: x.quantile(.025),
                   '0.975 quantile': lambda x: x.quantile(.975),
                   '0.16 quantile': lambda x: x.quantile(.16),
                   '0.84 quantile': lambda x: x.quantile(.84),
                   '0.25 quantile': lambda x: x.quantile(.25),
                   '0.75 quantile': lambda x: x.quantile(.75)}
    # Compute summary statistics for each strain component
    rows = []
    for c in cols_out:
        for k, fn in fn_from_key.items():
            rows.append({'component': c,
                         'statistic': k,
                         'value': fn(data[c])})
    return pd.DataFrame(rows)

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

def strainimg(df, field, bbox=None):
    """Create a strain image from a list of values.

    field := Column name in `df` containing the strain data to plot.

    bbox := Bounding box [xmin, xmax, ymin, ymax].  Values are
    inclusive.  Only pixels with coordinates on the boundary or in the
    interior of the bounding box are used for the strain image.  Hence,
    a bounding box of [-10.5, 10.5, 10.5, 20.5] is equivalent to [-10,
    10, 11, 20].

    """
    if bbox is None:
        bbox = [min(df['x']), max(df['x']),
                min(df['y']), max(df['y'])]

    bbox = clip_bbox_to_int(bbox)

    # Vic-2D indexes x and y from 0
    strainfield = np.empty((bbox[3] - bbox[2] + 1,
                            bbox[1] - bbox[0] + 1))
    strainfield.fill(np.nan)
    x = df['x'] - bbox[0]
    y = df['y'] - bbox[2]
    v = df[field]
    strainfield[[y, x]] = v
    return strainfield

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
    i_bb = np.array([[i_shp[0], i_shp[1]],
                     [0, i_shp[1]],
                     [0, 0],
                     [i_shp[0], 0]]).T
    x_aff_i = np.array([[0, 1],
                        [1, 0]])
    x_bb = np.dot(x_aff_i, i_bb)
    y_aff_x = basis
    y_bb = np.dot(y_aff_x, x_bb)
    y_max = np.ceil(np.max(y_bb, axis=1)).astype('int')
    y_min = np.floor(np.min(y_bb, axis=1)).astype('int')
    y_grid = np.mgrid[y_min[0]:y_max[0], y_min[1]:y_max[1]]
    j_shp = y_grid.shape[1:]

    i_transf = inv(x_aff_i) @ inv(y_aff_x) @ y_grid.reshape(2, -1)
    img_transf = map_coordinates(img, i_transf, cval=np.nan, order=order).reshape(j_shp)
    mask_transf = map_coordinates(img.mask, i_transf, cval=np.nan, order=0).reshape(j_shp)
    img_transf[mask_transf] = np.nan
    # Return to standard image convention
    return img_transf.swapaxes(0, 1)

def img(df, col, shp):
    """Convert a column in a Vic-2D csv export to an image.

    """
    # Note: Vic-2D indexes x and y from 0
    i = np.empty(shp)
    i.fill(np.nan)
    i[[df['y'], df['x']]] = df[col]
    return i

def read_strain_components(pth):
    """Read all strain component images from a Vic-2D csv file.

    """
    table = readv2dcsv(pth)
    if len(table) != 0:
        bbox = [np.min(table['x'].values), np.max(table['x'].values),
                np.min(table['y'].values), np.max(table['y'].values)]
        exx = mechana.vic2d.strainimg(table, 'exx', bbox)
        eyy = mechana.vic2d.strainimg(table, 'eyy', bbox)
        exy = mechana.vic2d.strainimg(table, 'exy', bbox)
    else:
        exx = []
        eyy = []
        exy = []
    components = {'exx': exx,
                  'eyy': eyy,
                  'exy': exy}
    return components

def plot_strains(csvpath):
    """Return three-panel strain fields figure from a Vic-2D .csv file.

    """
    df = mechana.vic2d.readv2dcsv(csvpath)

    # Find extent of region that has values
    xmin = min(df['x'])
    xmax = max(df['x'])
    ymin = min(df['y'])
    ymax = max(df['y'])

    ## Initialize figure
    fig = plt.figure(figsize=(6.0, 2.5), dpi=300, facecolor='w')
    ax1 = fig.add_subplot(131, aspect='equal')
    ax2 = fig.add_subplot(132, aspect='equal')
    ax3 = fig.add_subplot(133, aspect='equal')
    axes = [ax1, ax2, ax3]
    mpl.rcParams.update({'font.size': 10})

    ## Add the three strain plots
    fields = ['exx', 'eyy', 'exy']
    ctitles = ['$e_{xx}$', '$e_{yy}$','$e_{xy}$']
    for i, field in enumerate(fields):
        ## Plot strain image
        im = strainimg(df, field)
        ax = axes[i]
        cmin, cmax = np.percentile(df[field].values, [5, 95])
        extremum = max(abs(cmin), abs(cmax))
        leftextend = np.nanmin(im) < -extremum
        rightextend = np.nanmax(im) > extremum
        if leftextend and not rightextend:
            extend = 'min'
        elif rightextend and not leftextend:
            extend = 'max'
        elif leftextend and rightextend:
            extend = 'both'
        else:
            extend = 'neither'

        implot = ax.imshow(im, cmap="lab_diverging",
                           vmin=-extremum, vmax=extremum)

        ## Format axis
        ax.axis('off')
        ax.axis((xmin, xmax, ymin, ymax))
        ax.invert_yaxis()

        ## Add colorbar
        ticker = mpl.ticker.MaxNLocator(nbins=4)
        cbar = fig.colorbar(implot,
                            orientation='horizontal',
                            extend=extend,
                            ticks=ticker, ax=ax)
        cbar.set_label(ctitles[i], size=14)

        ## Set colorbar limits
        clim = max(abs(cmin), abs(cmax))
        cbar.set_clim((-clim, clim))

    ## Format figure
    plt.tight_layout()

    return fig

def plot_vic2d_data(simg, component, gimg=None, scale=None,
                    fig_width=5, fig_height=4, fig_fontsize=12,
                    cmap=None, norm=None,
                    basis=np.eye(2)):
    """Plot a strain field from a Vic-2D data table.

    """
    fig = plt.figure(figsize=(fig_width, fig_height),
                     frameon=False)
    ax = fig.add_subplot(111)
    ax.axis('off')

    limits = (np.nanpercentile(simg, 5),
              np.nanpercentile(simg, 95))
    if cmap is None and norm is None:
        cmap, norm = choose_cmap(limits)

    # Transform the strain field so it is aligned with provided
    # axes.
    simg = transform_image(simg, basis)

    # Plot photo
    if gimg is not None:
        gimg = transform_image(gimg, basis)
        aximg_gray = ax.imshow(gimg, cmap='gray', origin='lower')

    # Plot strain field
    aximg_strain = ax.imshow(simg, cmap=cmap, norm=norm, origin='lower')

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
        bars.add_artist(Rectangle((0,0), px_barw, px_barh, fc="black"))
        offsetbox = AnchoredOffsetbox(4, pad=0.1, borderpad=0.1,
                                      child=bars, frameon=False)
        ax.add_artist(offsetbox)

    ## Add colorbar
    if not np.all(simg[np.logical_not(np.isnan(simg))] == 0):
        ticker = mpl.ticker.MaxNLocator(nbins=5)
    else:
        ticker = mpl.ticker.FixedLocator([-1, 0, 1])
    cb = fig.colorbar(aximg_strain,
                      ticks=ticker, extend='both',
                      label=r'$E_{' + component[1:] + '}$',
                      orientation='horizontal')
    font = mpl.font_manager.FontProperties(size=fig_fontsize * 1.8)
    cb.ax.yaxis.label.set_font_properties(font)
    cb.ax.xaxis.label.set_font_properties(font)
    fig.tight_layout()
    return fig

def setup_vic2d(pth, imlist, imarchive, roi_xml=None):
    """Write a Vic-2D image list to a z2d file with the actual images."""
    # Make sure the output directory exists
    d, f = os.path.split(pth)
    fname, ext = os.path.splitext(f)
    dir_images = os.path.join(d, fname)
    if not os.path.exists(dir_images):
        os.makedirs(dir_images)
    if roi_xml is not None:
        # Write the z2d file to the output directory
        xml = replace_imlist_z2dxml(roi_xml,
            [os.path.join(fname, i) for i in imlist])
        with ZipFile(os.path.join(d, fname + '.z2d'), 'w') as f:
            f.writestr('project.xml', xml)
    # Write the image list to the output directory
    with open(os.path.join(d, fname + '.txt'), 'w') as f:
        for ln in imlist:
            f.write(fname + '/' + ln + '\n')
    # Write the images to the output directory
    for nm in imlist:
        with open(os.path.join(dir_images, nm), 'wb') as f:
            f.write(imarchive.read(nm))

def replace_imlist_z2dxml(xml, imlist):
    """Replace the <files> tag in Vic-2D XML with a new image list.

    The first image in `imlist` will be used as the reference image.
    Only the <files> tag and its children will be modified.

    """
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(BytesIO(xml), parser) # so pretty-printing works
    root = tree.getroot()
    # Remove the existing <files> element
    e = root.find('files')
    root.remove(e)
    # Build the new image list
    e_files = ET.Element('files', attrib={'lri': 'files'})
    ET.SubElement(e_files, 'reference').text = imlist[0]
    for i in imlist:
        ET.SubElement(e_files, 'deformed').text = i
    # Insert the new image list
    root.insert(0, e_files)
    return ET.tostring(tree, pretty_print = True, doctype='<!DOCTYPE vpml>')
