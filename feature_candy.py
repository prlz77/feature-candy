# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:43:13 2015

@author: prlz7
"""
# Allow access to command-line arguments
import sys
from caffe_reader import *
 
# Import the core and GUI elements of Qt
from PySide.QtCore import *
from PySide.QtGui import *
qt_app = QApplication(sys.argv)
import os
from os.path import expanduser
from mainwindow import Ui_MainWindow
 
class MainWindow(QMainWindow, Ui_MainWindow):
    ''' An example of PySide/PyQt absolute positioning; the main window
        inherits from QWidget, a convenient widget for an empty window. '''
 
    def __init__(self, NNReader):
        self.root_dir = expanduser("~")        
        self.NNR = NNReader     
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.connectWidgets()
        self.loaded = False
    def connectWidgets(self):
        self.selectModelButton.clicked.connect(self.selectModelButtonClicked)        
        self.selectWeightsButton.clicked.connect(self.selectWeightsButtonClicked)        
        self.selectMeanButton.clicked.connect(self.selectMeanButtonClicked)        
        self.loadModelButton.clicked.connect(self.loadModelButtonClicked)        
        self.graphicsView.wheelEvent = self.scale
        self.layersComboBox.activated[str].connect(self.layersComboBoxActivated)
        self.showNFilters.valueChanged[unicode].connect(self.showNFiltersActivated)
    def selectModelButtonClicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.root_dir)
        self.root_dir = os.path.dirname(fname)
        self.modelPathLabel.setText(fname)
    def selectWeightsButtonClicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.root_dir)
        self.root_dir = os.path.dirname(fname)
        self.weightsPathLabel.setText(fname) 
    def selectMeanButtonClicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.root_dir)
        self.root_dir = os.path.dirname(fname)
        self.meanPathLabel.setText(fname)       
    def loadModelButtonClicked(self):
        try:
            self.NNR.load(str(self.modelPathLabel.text()), str(self.weightsPathLabel.text()), str(self.meanPathLabel.text()))
            self.loadModelButton.setStyleSheet('QPushButton {background-color:green}')
            layer_names = self.NNR.get_layer_list()
            self.layersComboBox.insertItems(0, layer_names)
            self.loadInfoLabel.setText('Success.')
            self.loaded = True
        except Exception as e:
            self.loadInfoLabel.setText(str(e))
            self.loadModelButton.setStyleSheet('QPushButton {background-color:red}')
        
        layer_name = str(self.layersComboBox.currentText())
        num_filters = NNR.get_filters(layer_name).shape[0]
        
        self.showNFilters.setValue(num_filters)
        self.showNFilters.setMaximum(num_filters)
        
        self.paintFilters(layer_name)      

    def layersComboBoxActivated(self, text):
        num_filters = NNR.get_filters(text).shape[0]
        self.showNFilters.setValue(num_filters)
        self.showNFilters.setMaximum(num_filters)
        self.paintFilters(text)
    def showNFiltersActivated(self, value):
        self.paintFilters(str(self.layersComboBox.currentText()))
    def paintFilters(self, layer_name):
        if not self.loaded:
            return
        filters = NNR.get_filters(layer_name)[:self.showNFilters.value(), ...]
        sqfilters = NNR.vis_square(filters)
        sqfilters = (sqfilters * 255).astype('uint8')
        if len(sqfilters.shape) != 3:
            sqfilters = np.repeat(sqfilters[:,:,np.newaxis], 3, -1)
        image = QImage(sqfilters.data, sqfilters.shape[0], sqfilters.shape[1], (sqfilters.shape[1]*3), QImage.Format_RGB888)
        item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.resetTransform()
        self.graphicsView.fitInView(item, Qt.KeepAspectRatio)
        self.graphicsView.show()
    def scale(self, this):
        if this.delta() > 0:
            self.graphicsView.scale(1.25,1.25)
        else:
            self.graphicsView.scale(0.75,0.75)
    def run(self):
        # Show the form
        self.show()
        # Run the qt application
        qt_app.exec_()
 
# Create an instance of the application window and run it
NNR = CaffeReader()
app = MainWindow(NNR)
app.run()
