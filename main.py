
import os, sys, math, time, pdb
import librosa 

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QTableWidgetItem, QMessageBox, QWidget, QPushButton, 
    QHBoxLayout, QVBoxLayout, 
    QGroupBox, 
    QSizePolicy,
    QHeaderView, 
)

from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from GUI import Ui_MainWindow 
import cv2
import numpy as np 
import pandas as pd
import seaborn as sns

# for ecg signal processing
import wfdb 


import pyqtgraph as pg 


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)

        # this creates the UI and places all the widgets as placed in QtDesigner
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Title and dark theme
        self.setWindowTitle("Shazam meow")
        self.setStyleSheet('background-color: #333; color: white;')




        # ========================= Testing purposes ==================
        # self.plotWidget = pg.PlotWidget(self.ui.plotWidget)
        self.plotWidget = self.ui.graphicsView  # this one holds the plot 


        # api reference : https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html#pyqtgraph.PlotItem
        
        self.x_data = np.array([1,2,3,4,5])
        self.y_data = np.array([1,2,3,4,5])

        self.plotWidget.plot(self.x_data, self.y_data)  # Plot data
        self.plotWidget.setTitle("My Plot Title")  # Set title
        self.plotWidget.setLabel("left", "Y-Axis Label") 
        self.ui.btn_play.clicked.connect(self.plot_in_new_window)

    



    # testing how we can plot in a new window outside of the main window
    def plot_in_new_window(self): 
        pg.plot(self.x_data, np.array([1,1,1,1,1]))




if __name__ == '__main__':

    from warnings import filterwarnings
    filterwarnings("ignore", category=DeprecationWarning)

    # To avoid weird behaviors (smaller items, ...) on big resolution screens
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


    app = QApplication(sys.argv)
    app.setStyle("fusion")
    app.setWindowIcon(QtGui.QIcon(':icons/health.png'))

    def quit_app(): 
        print('Bye...')
        sys.exit(app.exec_())
    
    window = MyMainWindow()
    window.show()
    quit_app()







  