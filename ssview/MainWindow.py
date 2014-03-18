from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot
import pyqtgraph as pg
import numpy as np
import testdataIO
import os
from PIL import Image
import csv
from operator import itemgetter

from StressStrainVideoWidget import StressStrainVideoWidget

class MainWindow(QtGui.QMainWindow):

    # mwidg : StressStrainVideoWidget

    def __init__(self, parent=None):
        # Initialize the Qt parts of the UI
        super(MainWindow, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, 'MainWindow.ui'), self)
        # Initialize the python parts of the UI
        self.actionOpen.triggered.connect(self.open_file)
        self.actionQuit.triggered.connect(QtGui.qApp.quit)
        self.mwidg = StressStrainVideoWidget()
        self.setCentralWidget(self.mwidg)

    def open_file(self):
        # Load mechanical data
        fpath = str(QtGui.QFileDialog.getOpenFileName(self,
            'Open stress & strain csv file'))
        with open(fpath, 'rb') as f:
            reader = csv.reader(f, delimiter=",")
            reader.next() # skip header
            data = zip(*[(float(t), float(y), float(s)) 
                         for t, y, s in reader])
            time = data[0]
            stretch = data[1]
            stress = data[2]
        self.mwidg.update_mechdata(time, stretch, stress)
        # Load images
        ssdir = os.path.dirname(fpath)
        imdir = os.path.join(ssdir, "images")
        if not os.path.isdir(imdir):
            imdir = os.path.join(ssdir, "..", "images")
        imlist = self.load_images(imdir)
        d = zip(*imlist)
        self.mwidg.imstack.update_data(times=d[0],
                                       images=d[1])
        self.mwidg.signalDataChanged.emit()


    def load_images(self, imdir):
        """Load images into memory and find time of each frame.

        The times are calculated to be consistent with the stress &
        strain data file.  The reference image (typically defined as
        the frame of first motion) is assigned the time given in
        `ref_time.csv`; the remaining frame times are calculated from
        their file names.

        """
        # Load image names and times
        imlist = []
        for fname in os.listdir(imdir):
            if fname.endswith((".tiff", ".tif")):
                imlist.append(fname)
        # Load reference time
        fpath_reftime = os.path.join(imdir, "ref_time.csv")
        with open(fpath_reftime, 'rb') as f:
            reader = csv.reader(f)
            physical_reftime = float(reader.next()[0])
        # Apply offset such that reference frame has reference time
        fpath_imindex = os.path.join(imdir, "image_index.csv")
        refimgname = testdataIO.imindex_lookup(fpath_imindex, "reference")
        imtime = np.array(testdataIO.imtime(imlist))
        image_reftime = imtime[imlist.index(refimgname)]
        imtime = imtime - image_reftime + physical_reftime
        # Load images into memory
        images = []
        for fname in imlist:
            fpath = os.path.join(imdir, fname)
            images.append(np.array(Image.open(fpath)).T)
        return sorted(zip(imtime, images))


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
