from PyQt4 import QtGui, QtCore, uic
from PyQt4.QtCore import pyqtSlot, pyqtSignal
import pyqtgraph as pg
import numpy as np
import os
from PIL import Image
import csv
from operator import itemgetter

from data_gui import DataView

class MainWindow(QtGui.QMainWindow):

    # Variables
    currentfile = None

    # Gui objects
    data_view = None

    # Signals
    signalDataChanged = pyqtSignal()

    def __init__(self, parent=None):
        # Initialize the Qt parts of the UI
        super(MainWindow, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, 'MainWindow.ui'), self)
        # Initialize the python parts of the UI
        self.data_view = DataView()
        self.actionOpen.triggered.connect(self.open_file)
        self.actionQuit.triggered.connect(QtGui.qApp.quit)
        self.setCentralWidget(self.data_view)

    def open_file(self):
        # Load mechanical data
        fpath = str(QtGui.QFileDialog.getOpenFileName(self,
            'Open test data JSON file'))
        self.currentfile = fpath
        self.data_view.load_data(fpath)

        #self.mwidg.imstack.update_data(times=d[0],
        #                               images=d[1])        

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
