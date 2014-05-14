import csv, re, json, os
import numpy as np
import pandas as pd
import mechana
import matplotlib.image as mpimg
import pyqtgraph as pg

def debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt.

    """
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()

cmap_div = pg.ColorMap(pos=np.arange(0.0, 1.0, 0.1),
                       color=np.array([[0.0941, 0.3098, 0.6353, 1.0],
                                       [0.2745, 0.3882, 0.6824, 1.0],
                                       [0.4275, 0.6000, 0.8078, 1.0],
                                       [0.6275, 0.7451, 0.8824, 1.0],
                                       [0.8118, 0.8863, 0.9412, 1.0],
                                       [0.9451, 0.9569, 0.9608, 1.0],
                                       [0.9569, 0.8549, 0.7843, 1.0],
                                       [0.9725, 0.7216, 0.5451, 1.0],
                                       [0.8824, 0.5725, 0.2549, 1.0],
                                       [0.7333, 0.4706, 0.2118, 1.0],
                                       [0.5647, 0.3922, 0.1725, 1.0]]))
cmap_div_lut = cmap_div.getLookupTable()

def render_image(img, levels=None):
    isnan = np.isnan(img)
    extrema = np.percentile(img[~np.isnan(img)], (5, 95))
    absmax = np.abs(np.max(img))
    if absmax == 0.0:
        absmax = 1.0
    if levels is None:
        extremum = np.max(np.abs(extrema))
        levels = [-extremum, extremum]
    img_argb = np.nan_to_num(img)
    img_argb, b = pg.makeRGBA(np.nan_to_num(img),
                              levels=levels, lut=cmap_div_lut)
    img_argb[isnan] = 0
    return img_argb

class TestData:
    """Test data (mechanical data, images, strain fields).

    """
    # Mechanical data (arrays have 1:1 mapping)
    stretch = None
    stress = None
    time = None

    imagelist = None
    image_t0 = None

    strainfields = None # list of monochrome uint8 images
    strainfields_argb = None # list of rgba images
    fieldtimes = None

    def __init__(self, jsonfile):
        """Read test data from a test file.

        """
        datadir = os.path.dirname(jsonfile)
        with open(jsonfile, 'rb') as f:
            testdesc = json.load(f)
        # Read stress and strain
        ssfile = os.path.join(datadir, testdesc['stress_strain_file'])
        if ssfile is not None:
            data = pd.read_csv(ssfile)
            self.stress = data['Stress (Pa)'].values
            self.stretch = data['Stretch Ratio'].values
            self.time = data['Time (s)'].values
        # Read images
        imagelist = testdesc['images']
        if imagelist is not None:
            # Image list
            imagelist = [os.path.join(datadir, fp)
                         for fp in testdesc['images']]
            self.imagelist = imagelist
            # Image times
            imagenames = (os.path.basename(f) for f in imagelist)
            imagetimes = [mechana.images.image_time(nm)
                          for nm in imagenames]
            self.image_t0 = imagetimes[0]
            self.imagetimes = np.array(imagetimes) - self.image_t0
            # Image lookup table
            imageids = [mechana.images.image_id(f)
                        for f in imagelist]
            self.imagedict = dict(zip(imageids, imagelist))
        else:
            self.imagelist = None
        # Read strain fields
        vic2dfolder = testdesc['vic2d_folder']
        if vic2dfolder is not None:
            if self.image_t0 is None:
                raise Exception("Cannot match Vic-2D times "
                                "to mechanical test data "
                                "without an offset defined "
                                "for the image time codes.")
            vic2dfiles = mechana.vic2d.listcsvs(vic2dfolder)
            dfs = (mechana.vic2d.readv2dcsv(fp)
                   for fp in vic2dfiles)

            def get_fields(df):
                exx = mechana.vic2d.strainimg(df, 'exx')
                eyy = mechana.vic2d.strainimg(df, 'eyy')
                exy = mechana.vic2d.strainimg(df, 'exy')
                fields = {'exx': exx,
                          'eyy': eyy,
                          'exy': exy}
                return fields

            fieldlist = [get_fields(df) for df in dfs]
            self.strainfields = fieldlist

            # Get overall quantiles for each strain field
            fnames = ['exx', 'eyy', 'exy']
            extrema = {}
            for fn in fnames:
                ims = (fd[fn] for fd in self.strainfields)
                ims = (im[~np.isnan(im)] for im in ims)
                l = (max(np.abs(np.percentile(im, (5, 95))))
                     for im in ims)
                extrema[fn] = max(l)

            def render_field_dict(fields, extrema):
                fields_rgba = dict()
                for k in fields:
                    levels = (-extrema[k], extrema[k])
                    fields_rgba[k] = render_image(fields[k],
                                                  levels=levels)
                return fields_rgba

            self.strainfields_argb = [render_field_dict(fd, extrema)
                                      for fd in fieldlist]

            csvnames = (os.path.basename(f) for f in vic2dfiles)
            fieldtimes = [mechana.images.image_time(nm)
                          for nm in csvnames]
            self.fieldtimes = np.array(fieldtimes) - self.image_t0

    def stress_at(self, t):
        """Return stress at time t.

        """
        return np.interp(t, self.time, self.stress)

    def stretch_at(self, t):
        """Return stretch at time t.

        """
        return np.interp(t, self.time, self.stretch)

    def strainfields_at(self, t):
      """Return strain fields at time t.

      """
      idx = np.argmin(np.abs(self.fieldtimes - t))
      fieldtime = self.fieldtimes[idx]
      fields = self.strainfields[idx]
      fields_argb = self.strainfields_argb[idx]
      return fields, fields_argb, fieldtime

    def image_at(self, t):
        """Return image at time t.

        """
        idx = np.argmin(np.abs(self.imagetimes - t))
        imtime = self.imagetimes[idx]
        image = mpimg.imread(self.imagelist[idx])
        return image, imtime


def imtime(imlist):
    """Get timecode from list of test image filenames.

    The image filenames are assumed to be in the convention followed
    by the Elliott lab LabView camera capture VI:

    `cam#_index_timeinseconds.tiff`

    """
    if type(imlist) is str:
        imlist = [imlist]
    pattern = r"(?<=cam0_)([0-9]+)_([0-9]+\.?[0-9]+)(\.[A-Za-z]+)"
    t = []
    for s in imlist:
        timecode = re.search(pattern, s).group(2)
        t.append(float(timecode))
    return t

def imindex_lookup(csvpath, key):
    """Returns image name from image index.

    If `imagename` is not in the index, returns None.

    """
    with open(csvpath, 'r') as csvfile:
        for row in csv.reader(csvfile):
            if row[0] == key:
                return row[1]
    return None
