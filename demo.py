import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox,
    QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal

class FileSaverThread(QThread):
    finished = pyqtSignal()

    def __init__(self, file_path, save_dir):
        super().__init__()
        self.file_path = file_path
        self.save_dir = save_dir

    def run(self):
        try:
            # Create save directory if it doesn't exist
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            # Copy the file to the save directory
            shutil.copy(self.file_path, self.save_dir)
        except Exception as e:
            print("Error:", e)
        finally:
            self.finished.emit()

class AudioProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Label for file uploader
        self.file_label = QLabel('Upload a WAV file:')
        layout.addWidget(self.file_label)

        # Button to upload WAV file
        self.upload_button = QPushButton('Upload')
        self.upload_button.clicked.connect(self.uploadFile)
        layout.addWidget(self.upload_button)

        # Dropdown menu for processing options
        self.process_combo = QComboBox()
        self.process_combo.addItem('Baseline')
        self.process_combo.addItem('Pitch Shifted Baseline')
        layout.addWidget(self.process_combo)

        # Button to test processing
        self.test_button = QPushButton('Test')
        self.test_button.clicked.connect(self.testProcessing)
        layout.addWidget(self.test_button)

        self.setLayout(layout)
        self.setWindowTitle('WAV File Processor')
        self.show()

    def saveFileInBackground(self, save_dir):
        self.thread = FileSaverThread(self.file_path, save_dir)
        self.thread.finished.connect(self.saveFinished)
        self.thread.start()

    def saveFinished(self):
        label = self.file_path.split('/')[-1]
        self.file_label.setText(f'File: {label} saved successfully.')

    def uploadFile(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select a WAV File", "", "WAV Files (*.wav)", options=options
        )
        if filename:
            self.file_path = filename
            path = os.getcwd()
            self.saveFileInBackground(path + '/dev_data/raw/gearbox/output')

    def testProcessing(self):
        if hasattr(self, 'file_path'):
            process_option = self.process_combo.currentText()
            if process_option == 'Baseline':
                self.processBaseline()
            elif process_option == 'Pitch Shifted Baseline':
                self.processPitchShiftedBaseline()
        else:
            self.file_label.setText('Please upload a WAV file first.')

    def processBaseline(self):
        print('Baseline processing')

    def processPitchShiftedBaseline(self):
        print('Pitch shifted baseline processing')


def main():
    app = QApplication(sys.argv)
    window = AudioProcessor()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
