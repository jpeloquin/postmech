import os

import numpy as np

from PyQt5 import QtGui, QtCore, uic
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject
import pyqtgraph as pg

from datainput import TestData
from render import cmap_div, cmap_div_lut, render_strain

pg.setConfigOptions(foreground="w")
tick_font = QtGui.QFont().setPointSize(11)


def debug_trace():
    """Set a tracepoint in the Python debugger that works with Qt"""
    from PyQt4.QtCore import pyqtRemoveInputHook
    from ipdb import set_trace

    pyqtRemoveInputHook()
    set_trace()


# Create image displays
def _create_imview():
    vb = pg.ViewBox()
    vb.setAspectLocked(True)
    vb.invertY()
    imitem = pg.ImageItem()
    vb.addItem(imitem)
    return vb, imitem


class MarkerPlotWidget(pg.PlotWidget):
    marker = None

    def __init__(self, parent=None):
        super(MarkerPlotWidget, self).__init__(parent)
        self.marker = pg.InfiniteLine(0, movable=True)
        self.marker.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 200)))
        self.marker.setZValue(1)
        self.addItem(self.marker)
        self.marker.hide()

        for ax in self.plotItem.axes.items():
            ax[1]["item"].setTickFont(tick_font)


class DataView(QtGui.QWidget):
    """Qt widget for displaying plot time marker and video in sync."""

    # Data objects
    data = None  # currently loaded test data

    # Gui objects defined here
    # imitem # pyqtgraph.ImageItem for video feed

    # Plots (names defined in Qt Creator)
    # stretch_vs_time
    # stress_vs_time
    # stress_vs_stretch

    # imstack : ImageStack
    # gview : pyqtgraph.widgets.GraphicsView
    # sswidget : pyqtgraph.PlotWidget
    # ssitem : pyqtgraph.PlotCurveItem

    t = 0  # Current time

    signalDataChanged = pyqtSignal()
    signalMarkerMoved = pyqtSignal()

    def __init__(self, parent=None, parent_window=None):
        super(DataView, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, "DataView.ui"), self)

        # Default settings
        self.label_style = {"font-size": "14pt", "color": "white"}

        # Store references to other objects
        self.parent_window = parent_window
        # Format line plots
        self.stretch_vs_time.setLabel("left", text="Stretch Ratio", **self.label_style)
        self.stretch_vs_time.setLabel(
            "bottom", text="Time", units="s", **self.label_style
        )
        self.stress_vs_time.setLabel(
            "left", text="Stress", units="Pa", **self.label_style
        )
        self.stress_vs_time.setLabel(
            "bottom", text="Time", units="s", **self.label_style
        )
        self.stress_vs_stretch.setLabel(
            "left", text="Stress", units="Pa", **self.label_style
        )
        self.stress_vs_stretch.setLabel(
            "bottom", text="Stretch Ratio", **self.label_style
        )
        # Link time axes
        self.stretch_vs_time.setXLink(self.stress_vs_time)

        def link_views_xy(view1, view2):
            """Sets view1 to follow view2."""
            view1.setXLink(view2)
            view1.setYLink(view2)

        # Camera view
        self.camera_viewbox, self.camera_imitem = _create_imview()
        self.camera_plotitem = pg.PlotItem(viewBox=self.camera_viewbox)
        self.camera_plotitem.showAxis("left", show=False)
        self.camera_plotitem.showAxis("bottom", show=False)
        self.cameraview.setCentralItem(self.camera_plotitem)

        # Add strain field viewers
        self.components = ["exx", "eyy", "exy"]
        self.component_labels = {
            "exx": "e<sub>xx</sub>",
            "eyy": "e<sub>yy</sub>",
            "exy": "e<sub>xy</sub>",
        }
        self.strain_widget = {}
        self.strain_layout = {}
        self.strain_view = {}
        self.strain_vb = {}
        self.strain_imitem = {}
        self.strain_legend = {}
        for c in self.components:
            self.strain_view[c] = pg.GraphicsView()
            vb, imitem = _create_imview()
            self.strain_vb[c] = vb
            self.strain_imitem[c] = imitem
            self.strain_view[c].setCentralItem(vb)

            # Add color legend
            self.strain_legend[c] = ColorLegendWidget()
            self.strain_legend[c].setColormap(cmap_div)
            self.strain_legend[c].item.setImageItem(self.strain_imitem[c])
            # Set axis label
            axis = self.strain_legend[c].item.axis
            # e_xx, e_yy_, etc. look small, so we'll give these labels
            # a bigger font size
            s = {"font-size": "16pt", "color": "white"}
            axis.setLabel(self.component_labels[c], **s)
            # Set tick label style
            axis.setTickFont(tick_font)
            axis.setStyle(tickTextOffset=4)
            # Connect signal to update image when levels are changed
            sig = self.strain_legend[c].item.region.sigRegionChanged
            sig.connect(self.update_images)

            # Create layout and add to app
            self.strain_layout[c] = QtGui.QVBoxLayout()
            self.strain_layout[c].addWidget(self.strain_view[c])
            self.strain_layout[c].addWidget(self.strain_legend[c])
            self.strain_widget[c] = pg.GraphicsView()
            self.strain_widget[c].setLayout(self.strain_layout[c])
            self.strain_fields.layout().addWidget(self.strain_widget[c])

        # link axes of strain field plots
        link_views_xy(self.strain_vb["eyy"], self.strain_vb["exx"])
        link_views_xy(self.strain_vb["exy"], self.strain_vb["exx"])

        # Connect signals to slots for stress & stretch markers
        sig = self.stress_vs_stretch.marker.sigDragged
        sig.connect(self.on_stress_stretch_moved)
        sig = self.stress_vs_time.marker.sigDragged
        sig.connect(self.on_stress_time_moved)
        sig = self.stretch_vs_time.marker.sigDragged
        sig.connect(self.on_stretch_time_moved)

    def load_data(self, fpath):
        self.data = TestData.from_json(fpath)
        # Change title bar of main window
        if self.parent_window is not None:
            s = os.path.relpath(fpath, start=os.path.join(fpath, "../.."))
            self.parent_window.setWindowTitle(s)
        # Populate line plots with data
        self.stretch_vs_time.plot(x=self.data.time, y=self.data.stretch, antialias=True)
        self.stress_vs_time.plot(x=self.data.time, y=self.data.stress, antialias=True)
        self.stress_vs_stretch.plot(
            x=self.data.stretch, y=self.data.stress, antialias=True
        )
        self.stretch_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_stretch.marker.setPos(self.data.stretch[0])
        # Update images
        self.t = self.data.time[0]
        self.update_images()
        # Update color legends
        for c in self.components:
            if self.data.extrema[c] is not None:
                lim = self.data.extrema[c]
                self.strain_legend[c].item.setLimits((-lim, lim))
        # Update markers
        self.stress_vs_stretch.marker.setPos(self.data.stretch[0])
        self.stress_vs_time.marker.setPos(self.data.time[0])
        self.stretch_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_stretch.marker.show()
        self.stress_vs_time.marker.show()
        self.stretch_vs_time.marker.show()

        # self.stretch_vs_time.addItem(pg.PlotCurveItem())

    def unload_data(self):
        """Clear currently loaded data."""
        self.stretch_vs_time.clear()
        self.stress_vs_time.clear()
        self.stress_vs_stretch.clear()
        self.camera_imitem.setImage()
        self.exx_imitem.setImage()
        self.eyy_imitem.setImage()
        self.exy_imitem.setImage()

    @pyqtSlot()
    def on_stress_time_moved(self):
        """Move markers and update image plots to match current time."""
        self.t = self.stress_vs_time.marker.value()
        self.stretch_vs_time.marker.setValue(self.t)
        y = self.data.stretch_at(self.t)
        self.stress_vs_stretch.marker.setValue(y)
        self.update_images()

    @pyqtSlot()
    def on_stress_stretch_moved(self):
        """Move markers and update image plots to match current time."""
        y = self.stress_vs_time.marker.value()
        self.t = np.interp(y, self.data.stretch, self.data.time)
        self.stretch_vs_time.marker.setValue(self.t)
        self.stress_vs_time.marker.setValue(self.t)
        self.update_images()

    @pyqtSlot()
    def on_stretch_time_moved(self):
        """Move markers and update image plots to match current time."""
        self.t = self.stretch_vs_time.marker.value()
        self.stress_vs_time.marker.setValue(self.t)
        y = self.data.stretch_at(self.t)
        self.stress_vs_stretch.marker.setValue(y)
        self.update_images()

    def update_images(self):
        """Display images corresponding to time t"""
        # Camera frames
        if self.data.imagepaths is not None:
            image, mdata = self.data.image_at(self.t)
            self.camera_imitem.updateImage(image)
            self.camera_plotitem.setTitle(
                mdata["name"], size=self.label_style["font-size"]
            )
        # Strain fields
        if self.data.strainfields is not None:
            fields, fieldtime = self.data.strainfields_at(self.t)
            for c in self.components:
                img = render_strain(fields[c], levels=self.strain_legend[c].item.levels)
                self.strain_imitem[c].setImage(img)
                # Update histogram
                data = fields[c].reshape(-1)
                data = data[np.logical_not(np.isnan(data))]
                h = np.histogram(data, bins=500)
                self.strain_legend[c].item.histogram.setData(h[0], h[1][:-1])

    def update_mechdata(self, time, stretch, stress):
        self.mechtime = np.array(time)
        self.stretch = np.array(stretch)
        self.stress = np.array(stress)

        self.ssplot.setLabel("left", text="Stress", units="Pa")
        self.ssplot.setLabel("bottom", text="Stretch Ratio")

    def handle_data_changed(self):
        """Change UI in response to loading data.

        Specifically:
        1. Show the time marker if it isn't shown already
        2. Move the time marker to the reference image
        3. Restrict the time marker to the range for which images exist
        4. Show the images

        """
        if self.imstack.times is not None:
            # Show images
            self.refresh_imview()

    def load_images(self):
        #        from PyQt4.QtCore import pyqtRemoveInputHook
        #        import pdb; pyqtRemoveInputHook(); pdb.set_trace()
        # Show time marker
        self.tmark.show()
        # Move the time marker to the first image
        t0 = self.imstack.times[0]
        stretch0 = self.strain(t0)
        self.tmark.setPos(stretch0)
        # Restrict the marker motion
        t1 = self.imstack.times[-1]
        stretch1 = self.strain(t1)
        self.tmark.setBounds([stretch0, stretch1])
        self.vbox.autoRange()

    def refresh_imview(self):
        self.imitem.updateImage(self.current_image())
        self.ssplot.setTitle("Time = {}".format(self.t))

    def current_image(self):
        current_strain = self.tmark.value()
        self.t = self.time(current_strain)
        img = self.imstack.image(self.t)
        return img

    def time(self, strain):
        """Return time value for given strain"""
        return np.interp(strain, self.stretch, self.mechtime)

    def strain(self, time):
        """Return strain value for given time"""
        return np.interp(time, self.mechtime, self.stretch)


class ImageStack(QObject):
    # imlist : list of (time, image) tuples, where image is a numpy
    #          array
    times = None
    images = None
    ct = None  # current time

    signalDataChanged = pyqtSignal()

    def __init__(self, parent=None, times=None, images=None):
        super(ImageStack, self).__init__(parent)
        self.update_data(times=times, images=images)

    def update_data(self, times=None, images=None):
        """Replace image data

        time is a list of time values
        imlist is a list of 2-D images

        """
        if times:
            self.times = times
        if images:
            self.images = images
            if self.times is None:
                self.times = list(range(len(images)))
        if (self.times is not None) and (self.images is not None):
            # Sort the data by time, in case it isn't already sorted
            imlist = sorted(zip(self.times, self.images))
            d = zip(*imlist)
            self.times = d[0]
            self.images = d[1]
        self.signalDataChanged.emit()

    def image(self, time):
        if self.images is None:
            return None
        idx = min(enumerate(self.times), key=lambda tup: abs(tup[1] - time))[0]
        return self.images[idx]


class ColorLegendWidget(pg.GraphicsView):
    def __init__(self, parent=None, *args, **kargs):
        background = kargs.get("background", "default")
        super(ColorLegendWidget, self).__init__(
            parent, useOpenGL=False, background=background
        )
        self.item = ColorLegendItem(*args, **kargs)
        self.setCentralItem(self.item)

    def setColormap(self, cmap):
        self.item.setColormap(cmap)


class ColorLegendItem(pg.GraphicsWidget):
    """A Widget to show a color legend for an image.

    This Widget was inspired by HistogramLUTItem.  Compared to
    HistogramLUTItem, it can display the color legend horizontally.
    However, it does not provide any lookup table editing
    functionality, although the image levels can still be set.  The
    primary purpose of this Widget is simply to display the scale.

    """

    def __init__(self):
        super(ColorLegendItem, self).__init__()
        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)

        # Default parameters
        self.limits = (-1, 1)
        self.levels = (self.limits[0], self.limits[1])
        self.image = None  # Reference to linked ImageItem

        # Set up histogram and selector for range of colormap
        self.vb = pg.ViewBox(parent=self)
        self.region = SymmetricLinearRegionItem(
            [self.levels[0], self.levels[1]], pg.LinearRegionItem.Vertical
        )
        self.setLimits(self.limits)

        self.histogram = pg.PlotDataItem()
        # Both rotation and inversion are necessary to make the
        # histogram line up with self.axis
        self.histogram.rotate(-90)
        self.vb.invertY(True)
        self.histogram.setFillLevel(0.0)
        self.histogram.setFillBrush((100, 100, 200))

        self.vb.addItem(self.region)
        self.vb.addItem(self.histogram)

        self.axis = pg.AxisItem("bottom", linkView=self.vb, parent=self)
        self.axis.setRange(*self.limits)  # Just to get axis on manual
        # range mode
        self.colorbar = ColorBar()

        self.layout.addItem(self.colorbar, 0, 0)
        self.layout.addItem(self.vb, 1, 0)
        self.layout.addItem(self.axis, 2, 0)

        # Signals
        self.region.sigRegionChanged.connect(self.regionChanging)
        self.region.sigRegionChangeFinished.connect(self.regionChanged)

    def imageChanged(self):
        """Handle a change in image data."""
        if self.image is None:
            self.histogram.clear()

    def paint(self, p, *args):
        xlim = self.region.getRegion()
        p1 = self.vb.mapFromViewToItem(
            self, pg.Point(xlim[0], self.vb.viewRect().center().y())
        )
        p2 = self.vb.mapFromViewToItem(
            self, pg.Point(xlim[1], self.vb.viewRect().center().y())
        )
        colorbar_rect = self.colorbar.mapRectToParent(self.colorbar.colorbar.rect())
        pen = self.region.lines[0].pen
        p.setPen(pen)
        p.drawLine(p1, colorbar_rect.bottomLeft())
        p.drawLine(p2, colorbar_rect.bottomRight())

    def setColormap(self, cmap):
        """Set the colormap."""
        self.colorbar.setColorMap(cmap)

    def setImageItem(self, imitem):
        """Link the color scale to an ImageItem.

        The histogram in the ColorLegendItem will display data from
        the linked ImageItem, disregarding NaNs.  The image will also
        have its levels set according to the ColorLegendItem.

        """
        self.image = imitem
        self.image.sigImageChanged.connect(self.imageChanged)
        self.imageChanged()

    def setLimits(self, limits):
        """Set axis limits."""
        self.limits = tuple(limits)
        self.vb.setXRange(*self.limits)
        self.region.setBounds(self.limits)
        self.region.line_min.setValue(self.limits[0])
        self.region.line_max.setValue(self.limits[1])

    def regionChanging(self):
        """Handle levels as they are being changed.

        The levels will change symmetrically about 0.

        """
        self.levels = self.region.getRegion()
        # Force levels to be symmetric about 0
        # level = max(np.abs(self.levels))
        # self.levels = (-level, level)
        # self.region.blockLineSignal = True
        # self.region.lines[0].setValue(self.levels[0])
        # self.region.lines[1].setValue(self.levels[1])
        # self.region.blockLineSignal = False
        self.update()  # keeps lines from region boundaries to colorbar
        # corners from smearing or leaving artifacts

    def regionChanged(self):
        """Handle new levels."""
        pass


class ColorBar(pg.GraphicsWidget):
    """A plain old colorbar."""

    def __init__(self, *kargs, **kwargs):
        super(ColorBar, self).__init__()
        # Default parameters
        self.w = 200
        self.h = 30
        self.colormap = pg.ColorMap(
            pos=np.array([0, 1.0]),
            color=np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]),
        )

        self.setMaximumHeight(self.h)
        self.setMaximumWidth(16777215)
        self.colorbar = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, self.w, self.h))
        self.colorbar.setParentItem(self)

        self.gradient = self.getGradient()
        self.colorbar.setBrush(QtGui.QBrush(self.gradient))

    def getGradient(self):
        """Return a QLinearGradientObject."""
        g = pg.QtGui.QLinearGradient(0.0, 0.0, self.w, self.h)
        pos, color = self.colormap.getStops(pg.ColorMap.BYTE)
        color = [QtGui.QColor(*x) for x in color]
        g.setStops([a for a in zip(pos, color)])
        return g

    def resizeEvent(self, ev):
        """Fired by Qt when widget is (re)sized."""
        self.w = self.width()
        # self.width is from pg.GraphicsWidget
        self.colorbar.setRect(0.0, 0.0, self.w, self.h)
        self.updateGradient()

    def setColorMap(self, cmap):
        """Recreate the gradient with a new color map."""
        self.colormap = cmap
        self.updateGradient()

    def updateGradient(self):
        """Update the displayed gradient.

        It is appropriate to call this after (1) resizing, in which
        case the gradient must be painted over a different region or
        (2) the underlying colormap changing.

        """
        self.gradient = self.getGradient()
        self.colorbar.setBrush(QtGui.QBrush(self.gradient))


class SymmetricLinearRegionItem(pg.LinearRegionItem):
    """A LinearRegion item that is symmetric about a value."""

    def __init__(self, *args, **kargs):
        super(SymmetricLinearRegionItem, self).__init__()

        # Separate low and high boundary lines
        for l in self.lines:
            l.sigPositionChanged.disconnect(self.lineMoved)
        levels = [(l.value(), l) for l in self.lines]
        self.line_min = min(levels)[1]
        self.line_max = max(levels)[1]

        self.line_min.sigPositionChanged.connect(self.lineMinMoved)
        self.line_max.sigPositionChanged.connect(self.lineMaxMoved)

    def lineMinMoved(self):
        # Move line_max to match
        self.line_max.setValue(-self.line_min.value())
        self.lineMoved()

    def lineMaxMoved(self):
        # Move line_min to match
        self.line_min.setValue(-self.line_max.value())
        self.lineMoved()
