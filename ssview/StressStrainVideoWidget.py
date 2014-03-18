from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot, pyqtSignal, QObject
import pyqtgraph as pg
import numpy as np
import os

class StressStrainVideoWidget(QtGui.QWidget):
    """Qt widget for displaying plot time marker and video in sync.


    """
    # imstack : ImageStack
    # gview : pyqtgraph.widgets.GraphicsView
    # sswidget : pyqtgraph.PlotWidget
    # ssitem : pyqtgraph.PlotCurveItem

    mechtime = None
    stretch = None
    stress = None

    t = 0 # Current time

    signalDataChanged = pyqtSignal()
    signalTMarkerMoved = pyqtSignal()

    def __init__(self, parent=None):
        super(StressStrainVideoWidget, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, 'StressStrainVideoWidget.ui'), self)
        # Adjust layout
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        # Attach time marker to stress-strain plot
        self.tmark = pg.InfiniteLine(1, movable=True)
        self.tmark.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 200)))
        self.tmark.setZValue(1)
        self.ssplot.addItem(self.tmark)
        self.tmark.hide()
        # Create stress-strain plot item
        self.ssitem = pg.PlotCurveItem()
        self.ssplot.addItem(self.ssitem)
        # Create image item
        self.imitem = pg.ImageItem()
        self.vbox = pg.ViewBox()
        self.gview.setCentralItem(self.vbox)
        self.vbox.setAspectLocked(True)
        self.vbox.invertY()
        self.vbox.addItem(self.imitem)
        self.imstack = ImageStack()
        # Connect signals and slots
        self.signalDataChanged.connect(self.handle_data_changed)
        self.tmark.sigPositionChanged.connect(self.refresh_imview)
        self.imstack.signalDataChanged.connect(self.load_images)

    def update_mechdata(self, time, stretch, stress):
        self.mechtime = np.array(time)
        self.stretch = np.array(stretch)
        self.stress = np.array(stress)
        self.ssplot.plot(x=self.stretch, y=self.stress)
#        labelStyle = {'font-size': '14px'}
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
