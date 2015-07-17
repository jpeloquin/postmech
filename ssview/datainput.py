import numpy as np
import pyqtgraph as pg

from mechana.analysis import MechanicalTest

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

class TestData(MechanicalTest):
    """Test data (mechanical data, images, strain fields).

    """

    def __init__(self, *args):
        super(TestData, self).__init__(*args)

        # Get overall quantiles for each strain field
        fnames = ['exx', 'eyy', 'exy']
        extrema = {}
        for fn in fnames:
            ims = (fd[fn] for fd in self.strainfields)
            ims = (im[~np.isnan(im)] for im in ims)
            l = (max(np.abs(np.percentile(im, (5, 95))))
                 for im in ims)
            extrema[fn] = max(l)

        # Render strainfields for quick display
        self.strainfields_argb = [self._render_field_dict(fd, extrema)
                                  for fd in self.strainfields]

    def _render_field_dict(self, fields, extrema):
        fields_rgba = dict()
        for k in fields:
            levels = (-extrema[k], extrema[k])
            fields_rgba[k] = render_image(fields[k],
                                              levels=levels)
        return fields_rgba

    def strainfields_at(self, t):
        """Return strain fields at time t.

        """
        idx = np.argmin(np.abs(self.fieldtimes - t))
        fieldtime = self.fieldtimes[idx]
        fields = self.strainfields[idx]
        fields_argb = self.strainfields_argb[idx]
        return fields, fields_argb, fieldtime
