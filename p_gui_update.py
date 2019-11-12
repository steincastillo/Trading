#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:33:57 2019

@author: Stein
"""
#############
# Libraries
#############
import pandas as pd
import yaml
import stock_func as stock
from PyQt5 import QtWidgets, uic
import sys
 

# Set UI file
qtCreatorFile = "p_gui_update.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

#app = QtWidgets.QApplication([])
 
#win = uic.loadUi("p_gui_update.ui") #specify the location of your .ui file
 
#############
# Classes
#############
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        self.fileName = ''
        self.symbols = []
        self.predictions = {}
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        # Define a dialog box to display error mesages
        self.errMsg = QtWidgets.QErrorMessage()

        # Add status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        # Connect widgets
        self.btn_portfolio.clicked.connect(self.openEvent)  # Select portfolio file (YAML)
        self.btn_close.clicked.connect(self.closeEvent)     # Close the application
        self.btn_update.clicked.connect(self.updatePrices)  # Update stock prices 

    # Define function to process inteface events

    def closeEvent(self):
        # Close the application
        app.quit()
    
    def openEvent(self):
        # Select portfolio file (YAML)
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getSaveFileName()", "","YAML files (*.yml)", options=options)
        if fileName:
            # Read YAML file
            conf = yaml.load(open(fileName))
            # Unpack configuration from YAML file
            port_name = conf['general']['port_name']
            port_index = conf['general']['index']
            port_unit = conf['general']['unit']
            port_rfr = conf['general']['rfr']
            # Display the values on the GUI
            self.lin_file.setText(fileName)
            self.lin_portfolio.setText(port_name)
            self.lin_benchmark.setText(port_index)

            # Unpack portfolio symbols
            self.symbols = []
            stocks = conf['stocks']
            for s in stocks:
                sym = [*s.keys()]
                self.symbols.append(sym[0])
            # Add index symbol
            self.symbols.append(port_index)
            # Display symbols in list widget
            for s in self.symbols:
                self.list_symbols.addItem(s)

    def updatePrices(self):
        # Unpack the dates
        start_date = self.date_startDate.date().toPyDate()
        end_date = self.date_endDate.date().toPyDate()
        # Update the stock pricing information
        if len(self.symbols) > 0:
            for s in self.symbols:
                self.statusBar.showMessage('Retrieveing OHLC data for: {}'.format(s[0]), 1000)
                stock.create_csv(s, start_date, end_date)
            self.statusBar.showMessage('Process Complete!', 2000)
            QtWidgets.QMessageBox.about(self, 'Process complete', 'Update Sucessful!')
        else:
            # No portfolio selected
            self.errMsg.showMessage('Select a valid portfolio')
        
#############
# Main loop
#############
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
