import numpy as np
import pyqtgraph as pg

cmap_div = pg.ColorMap(pos=np.linspace(0.0, 1.0, 11),
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
cmap_div_lut = cmap_div.getLookupTable(nPts=256).astype(np.uint8)

def render_strain(strain, levels=None):
    """Render a strain field as an ARGB image.

    """
    isnan = np.isnan(strain)
    absmax = np.abs(np.max(strain))
    if absmax == 0.0:
        absmax = 1.0
    if levels is None:
        extrema = np.percentile(strain[~isnan], (5, 95))
        extremum = np.max(np.abs(extrema))
        levels = [-extremum, extremum]
    # strain[isnan] = 0
    strain_argb, b = pg.makeRGBA(strain, levels=levels, lut=cmap_div_lut)
    strain_argb = strain_argb * np.logical_not(np.expand_dims(isnan, 2))
    return strain_argb
