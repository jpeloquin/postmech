from PyQt5 import QtGui, QtCore, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import pyqtgraph as pg
import numpy as np
import os, sys
from PIL import Image
import csv
from operator import itemgetter

from data_gui import DataView


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        # Initialize the Qt parts of the UI
        super(MainWindow, self).__init__(parent)
        codedir = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi(os.path.join(codedir, "MainWindow.ui"), self)

        # Variables
        currentfile = None

        # Gui objects
        data_view = None

        # Signals
        signalDataChanged = pyqtSignal()

        # Initialize the python parts of the UI
        self.data_view = DataView(parent_window=self)
        self.actionOpen.triggered.connect(self.ui_open)
        self.actionQuit.triggered.connect(QtWidgets.qApp.quit)
        self.setCentralWidget(self.data_view)

    def ui_open(self):
        """Get file path from user and open data listed in it."""
        msg = "Open test data JSON file"
        pth, _filter = QtGui.QFileDialog.getOpenFileName(self, msg)
        self.load_data(pth)

    def load_data(self, pth):
        """Load data listed in a .json file."""
        self.currentfile = pth
        self.data_view.load_data(pth)


def main(pth_data=None):
    win = MainWindow()
    win.show()
    if pth_data is not None:
        win.load_data(pth_data)
    app.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 1:
        pth_data = sys.argv[1]
    else:
        pth_data = None
    main(pth_data=pth_data)
