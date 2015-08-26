import numpy as np
import pyqtgraph as pg

from mechana.analysis import MechanicalTest
from render import render_strain

def debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt.

    """
    from PyQt4.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()

class TestData(MechanicalTest):
    """Test data (mechanical data, images, strain fields).

    """
    def __init__(self, *args, **kargs):
        super(TestData, self).__init__(*args, **kargs)

        # Get overall quantiles for each strain field
        components = ['exx', 'eyy', 'exy']
        self.extrema = {}
        for c in components:
            ims = (fd[c] for fd in self.strainfields)
            ims = (im[~np.isnan(im)] for im in ims)
            l = (max(np.abs(np.percentile(im, (5, 95))))
                 for im in ims)
            self.extrema[c] = max(l)

        # Render strainfields for quick display
        self.strainfields_argb = [self._render_field_dict(fd, self.extrema)
                                  for fd in self.strainfields]

    def _render_field_dict(self, fields, extrema):
        fields_rgba = dict()
        for k in fields:
            levels = (-extrema[k], extrema[k])
            fields_rgba[k] = render_strain(fields[k],
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
