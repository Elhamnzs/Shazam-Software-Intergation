
import os, sys, math, time, pdb
import librosa 
import sounddevice as sd
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
        #this comment is for plot_in_new_window
        #self.ui.btn_play.clicked.connect(self.plot_in_new_window)
        self.plotWidget = self.ui.graphicsView  # this one holds the plot

        self.x_data = np.array([1, 2, 3, 4, 5])
        self.y_data = np.array([1, 2, 3, 4, 5])

        self.plotWidget.plot(self.x_data, self.y_data)  # Plot data
        self.plotWidget.setTitle("My Plot Title")  # Set title
        self.plotWidget.setLabel("left", "Y-Axis Label")

        # Connect the play button to both progress bar and plotting in a new window
        self.ui.btn_play.clicked.connect(self.start_progress_bar)

        # Timer for progress bar
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)

        self.progress_duration = 13.4 * 1000  # 15 seconds in milliseconds
        self.progress_step = 100 / (self.progress_duration / 100)  # Calculate step size
        self.current_progress = 0

        self.ui.btn_play.clicked.connect(self.start_recording_and_visualization)

        self.sample_rate = 44100  # Standard audio sample rate
        self.recording_duration = 15  # Duration in seconds
        self.audio_data = np.array([])  # Placeholder for recorded audio data

        # Timer for real-time plotting
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)

        # Variables for real-time plot
        self.plot_x = np.array([])  # Time axis
        self.plot_y = np.array([])  # Amplitude

    def start_recording_and_visualization(self):
        # Reset variables
        self.audio_data = np.array([])
        self.plot_x = np.array([])
        self.plot_y = np.array([])

        # Start the progress bar
        self.start_progress_bar()

        # Start recording in a separate thread to avoid UI blocking
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback
        )
        self.audio_stream.start()

        # Start the real-time plot update
        self.plot_timer.start(100)  # Update every 100 ms

        # Stop recording and plotting after 15 seconds
        QTimer.singleShot(self.recording_duration * 1000, self.stop_recording)

    def audio_callback(self, indata, frames, time, status):
        """Callback to handle incoming audio data."""
        if status:
            print(f"Audio stream status: {status}")
        self.audio_data = np.append(self.audio_data, indata[:, 0])  # Append the new audio data

    def update_plot(self):
        """Update the real-time plot with new audio data."""
        if len(self.audio_data) > 0:
            # Get the latest audio data to plot
            self.plot_y = self.audio_data[-self.sample_rate:]  # Last 1 second of data
            self.plot_x = np.linspace(0, len(self.plot_y) / self.sample_rate, len(self.plot_y))

            # Clear the previous plot and replot
            self.plotWidget.clear()
            self.plotWidget.plot(self.plot_x, self.plot_y, pen='y')  # Yellow waveform

    def stop_recording(self):
        """Stop recording and plotting."""
        self.audio_stream.stop()
        self.plot_timer.stop()
        print("Recording stopped!")

        # Function to start the progress bar

    def start_progress_bar(self):
        self.current_progress = 0
        self.ui.progressBar.setValue(self.current_progress)
        self.progress_timer.start(100)  # Update every 100ms

        # Continue with existing functionality
        #and here where I deactivate them

        #self.plot_in_new_window()

        # Function to update progress bar

    def update_progress_bar(self):
        self.current_progress += self.progress_step
        self.ui.progressBar.setValue(int(self.current_progress))

        # Stop the timer when progress reaches 100%
        if self.current_progress >= 100:
            self.progress_timer.stop()

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







  