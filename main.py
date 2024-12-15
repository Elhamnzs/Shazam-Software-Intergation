import sys
import sounddevice as sd
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QDialog, QStyleFactory
)
from UI.GUI import Ui_MainWindow 
from UI.infoDialog import Ui_Dialog

import numpy as np 
import pyqtgraph as pg 


class InfoDialog(QDialog): 
    def __init__(self, song_name:str):
        super().__init__()
        self.ui = Ui_Dialog()  
        self.ui.setupUi(self)

        self.ui.label_2.setText(song_name)
        self.ui.buttonBox.accepted.connect(self.accept)  # Close on button click
        self.setStyleSheet('background-color: #333; color: white;')
        self.setStyle(QStyleFactory.create('Fusion'))



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):

        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Shazam meow")
        self.setStyleSheet('background-color: #333; color: white;')



        self.plotWidget = self.ui.graphicsView  
 

        # Connect the play button to both progress bar and plotting the audio waves
        # self.ui.btn_play.clicked.connect(self._start_progress_bar)
        self.ui.btn_play.clicked.connect(self.start_recording_and_visualization)



        # Timer for progress bar
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)

        self.progress_duration = 13.4 * 1000  # 15 seconds in milliseconds
        self.progress_step = 100 / (self.progress_duration / 100)  # Calculate step size
        self.current_progress = 0

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
        self._start_progress_bar()

        # Start recording in a separate thread to avoid UI blocking (It is non blocking)
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback
        )
        self.audio_stream.start()

        # Start the real-time plot update
        self.plot_timer.start(100)  # Update every 100 ms

        # Stop recording and plotting after 15 seconds
        QTimer.singleShot(self.recording_duration * 1000, self._stop_recording)

    def _audio_callback(self, indata, frames, time, status):
        """Callback to handle incoming audio data."""
        if status:
            print(f"Audio stream status: {status}")
        self.audio_data = np.append(self.audio_data, indata[:, 0])  # Append the new audio data

    def _stop_recording(self):
        """Stop recording and plotting."""
        self.audio_stream.stop()
        self.plot_timer.stop()

        print("Recording stopped!")


        print('starting processing ....')
        # maybe start processin gthe stuff here in something 


        # maybe link this to the signal at the end 
        self._show_dialog('bob sinclar')

        # Function to start the progress bar

    def _start_progress_bar(self):
        self.current_progress = 0
        self.ui.progressBar.setValue(self.current_progress)
        self.progress_timer.start(100)  # Update every 100ms

        # Continue with existing functionality
        #and here where I deactivate them

        #self.plot_in_new_window()

        # Function to update progress bar



    # this is connected to timer for updating the plot
    def update_plot(self):
        """Update the real-time plot with new audio data."""
        if len(self.audio_data) > 0:
            # Get the latest audio data to plot
            self.plot_y = self.audio_data[-self.sample_rate:]  # Last 1 second of data
            self.plot_x = np.linspace(0, len(self.plot_y) / self.sample_rate, len(self.plot_y))

            # Clear the previous plot and replot
            self.plotWidget.clear()
            self.plotWidget.plot(self.plot_x, self.plot_y, pen='y')  # Yellow waveform

    def update_progress_bar(self):
        self.current_progress += self.progress_step
        self.ui.progressBar.setValue(int(self.current_progress))

        # Stop the timer when progress reaches 100%
        if self.current_progress >= 100:
            self.progress_timer.stop()



    # for displaying the matched song in a new dialgo
    def _show_dialog(self, song_name): 
        dialog = InfoDialog(song_name=song_name)
        dialog.exec_()  


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


  