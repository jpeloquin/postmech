import os

import numpy as np

from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QObject
import pyqtgraph as pg

from datainput import TestData
from render import cmap_div, cmap_div_lut

def debug_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook
    from ipdb import set_trace
    pyqtRemoveInputHook()
    set_trace()

class MarkerPlotWidget(pg.PlotWidget):
    marker = None

    def __init__(self, parent=None):
        super(MarkerPlotWidget, self).__init__(parent)
        self.marker = pg.InfiniteLine(0, movable=True)
        self.marker.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 200)))
        self.marker.setZValue(1)
        self.addItem(self.marker)
        self.marker.hide()

class DataView(QtGui.QWidget):
    """Qt widget for displaying plot time marker and video in sync.

    """
    # Data objects
    data = None # currently loaded test data

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

    t = 0 # Current time

    signalDataChanged = pyqtSignal()
    signalMarkerMoved = pyqtSignal()

    def __init__(self, parent=None, parent_window=None):
        super(DataView, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, 'DataView.ui'), self)
        # Store references to other objects
        self.parent_window = parent_window
        # Format line plots
        self.stretch_vs_time.setLabel('left', text='Stretch Ratio')
        self.stretch_vs_time.setLabel('bottom', text='Time', 
                                      units='s')
        self.stress_vs_time.setLabel('left', text='Stress', units='Pa')
        self.stress_vs_time.setLabel('bottom', text='Time', units='s')
        self.stress_vs_stretch.setLabel('left',text='Stress',
                                        units='Pa')
        self.stress_vs_stretch.setLabel('bottom', text='Stretch Ratio')
        # Link time axes
        self.stretch_vs_time.setXLink(self.stress_vs_time)

        # Create image displays
        def create_imview():
            vb = pg.ViewBox()
            vb.setAspectLocked(True)
            vb.invertY()
            imitem = pg.ImageItem()
            vb.addItem(imitem)
            return vb, imitem
        def link_views_xy(view1, view2):
            """Sets view1 to follow view2."""
            view1.setXLink(view2)
            view1.setYLink(view2)
        # Camera view
        self.camera_viewbox, self.camera_imitem = create_imview()
        self.camera_plotitem = pg.PlotItem(viewBox=self.camera_viewbox)
        self.camera_plotitem.showAxis('left', show=False)
        self.camera_plotitem.showAxis('bottom', show=False)
        self.cameraview.setCentralItem(self.camera_plotitem)
        # exx
        self.exx_viewbox, self.exx_imitem = create_imview()
        self.exxview.setCentralItem(self.exx_viewbox)
        # eyy
        self.eyy_viewbox, self.eyy_imitem = create_imview()
        self.eyyview.setCentralItem(self.eyy_viewbox)
        # exy
        self.exy_viewbox, self.exy_imitem = create_imview()
        self.exyview.setCentralItem(self.exy_viewbox)
        # link axes of strain field plots
        link_views_xy(self.eyy_viewbox, self.exx_viewbox)
        link_views_xy(self.exy_viewbox, self.exx_viewbox)
        # Add colorbar to strain field plots
        self.color_widget = pg.HistogramLUTWidget(image=self.exx_imitem)
        self.color_widget.gradient.setColorMap(cmap_div)
        self.field_layout.addWidget(self.color_widget)
        # Add different color scale legend
        self.color_legend_item = pg.GradientLegend((10, 100), (10, 30))
        self.color_legend_item.setParentItem(self.exx_viewbox)
        # Connect signals to slots for marker  
        self.stress_vs_stretch.marker.sigDragged.connect(self.on_stress_stretch_moved)
        self.stress_vs_time.marker.sigDragged.connect(self.on_stress_time_moved)
        self.stretch_vs_time.marker.sigDragged.connect(self.on_stretch_time_moved)

    def load_data(self, fpath):
        self.data = TestData(fpath)
        # Change title bar of main window
        if self.parent_window is not None:
            s = os.path.relpath(fpath,
                                start=os.path.join(fpath, '../..'))
            self.parent_window.setWindowTitle(s)
        # Populate line plots with data
        self.stretch_vs_time.plot(x=self.data.time,
                                  y=self.data.stretch,
                                  antialias=True)
        self.stress_vs_time.plot(x=self.data.time,
                                 y=self.data.stress,
                                 antialias=True)
        self.stress_vs_stretch.plot(x=self.data.stretch,
                                    y=self.data.stress,
                                    antialias=True)
        self.stretch_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_stretch.marker.setPos(self.data.stretch[0])
        # Update images
        self.t = self.data.time[0]
        self.update_images()
        # Update markers
        self.stress_vs_stretch.marker.setPos(self.data.stretch[0])
        self.stress_vs_time.marker.setPos(self.data.time[0])
        self.stretch_vs_time.marker.setPos(self.data.time[0])
        self.stress_vs_stretch.marker.show()
        self.stress_vs_time.marker.show()
        self.stretch_vs_time.marker.show()
        
        # self.stretch_vs_time.addItem(pg.PlotCurveItem())

    def unload_data(self):
        """Clear currently loaded data.

        """
        self.stretch_vs_time.clear()
        self.stress_vs_time.clear()
        self.stress_vs_stretch.clear()
        self.camera_imitem.setImage()
        self.exx_imitem.setImage()
        self.eyy_imitem.setImage()
        self.exy_imitem.setImage()

    @pyqtSlot()
    def on_stress_time_moved(self):
        """Move markers and update image plots to match current time.

        """
        self.t = self.stress_vs_time.marker.value()
        self.stretch_vs_time.marker.setValue(self.t)
        y = self.data.stretch_at(self.t)
        self.stress_vs_stretch.marker.setValue(y)
        self.update_images()

    @pyqtSlot()
    def on_stress_stretch_moved(self):
        """Move markers and update image plots to match current time.

        """
        y = self.stress_vs_time.marker.value()
        self.t = np.interp(y, self.data.stretch, self.data.time)
        self.stretch_vs_time.marker.setValue(self.t)
        self.stress_vs_time.marker.setValue(self.t)
        self.update_images()

    @pyqtSlot()
    def on_stretch_time_moved(self):
        """Move markers and update image plots to match current time.

        """
        self.t = self.stretch_vs_time.marker.value()
        self.stress_vs_time.marker.setValue(self.t)
        y = self.data.stretch_at(self.t)
        self.stress_vs_stretch.marker.setValue(y)
        self.update_images()

    def update_images(self):
        """Display images corresponding to time t

        """
        # Camera frames
        if self.data.imagepaths is not None:
            image, mdata = self.data.image_at(self.t)
            self.camera_imitem.updateImage(image)
            self.camera_plotitem.setTitle(mdata['name'])
        # Strain fields
        if self.data.strainfields is not None:
            fields, fields_rgba, fieldtime = self.data.strainfields_at(self.t)
            self.exx_imitem.setImage(fields['exx'])
            self.eyy_imitem.setImage(fields_rgba['eyy'])
            self.exy_imitem.setImage(fields_rgba['exy'])

    def update_mechdata(self, time, stretch, stress):
        self.mechtime = np.array(time)
        self.stretch = np.array(stretch)
        self.stress = np.array(stress)

        # labelStyle = {'font-size': '14px'}
        self.ssplot.setLabel('left', text='Stress', 
                             units='Pa')
        self.ssplot.setLabel('bottom', text='Stretch Ratio')

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
        """Return time value for given strain

        """
        return np.interp(strain, self.stretch, self.mechtime)

    def strain(self, time):
        """Return strain value for given time

        """
        return np.interp(time, self.mechtime, self.stretch)

class ImageStack(QObject):

    # imlist : list of (time, image) tuples, where image is a numpy
    #          array
    times = None
    images = None
    ct = None # current time

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
        idx = min(enumerate(self.times),
                  key=lambda tup: abs(tup[1]-time))[0]
        return self.images[idx]

class ColorLegendWidget(pg.GraphicsView):

    def __init__(self, parent=None, *args, **kargs):
        background = kargs.get('background', 'default')
        GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        # self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        # self.setMinimumHeight(95)

    
class ColorLegendItem(GraphicsWidget):
    """A Widget to show a color legend for an image.

    This Widget was inspired by HistogramLUTItem.  Compared to
    HistogramLUTItem, it can display the color legend horizontally.
    However, it does not provide any lookup table editing
    functionality, although the image levels can still be set.  The
    primary purpose of this Widget is simply to display the scale.

    """
    
