# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 00:11:43 2019

@author: stein (contact@steincastillo.com)

usage: python predict.py
"""

#############
# Libraries
#############
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from pandas_datareader import data as web
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV as rcv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Define UI
qtCreatorFile = "predict.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

#############
# Classes
#############
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        self.fileName = ''
        self.predictions = {}
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        # Add status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusBar)

        # Connect widgets
        self.bAdd.clicked.connect(self.add)
        self.bDelete.clicked.connect(self.delete)
        self.bDeleteAll.clicked.connect(self.delall)
        self.bPredict.clicked.connect(self.predict)
        self.bClose.clicked.connect(self.closeEvent)
        self.bSave.clicked.connect(self.saveEvent)

    def closeEvent(self):
        # Close the application
        app.quit()

    def add(self):
        # Add ticker to the list
        if len(self.lSymbol.text()) !=0:
            self.lTickers.addItem(self.lSymbol.text().upper())
        self.lSymbol.setText('')
        self.lSymbol.setFocus()
    
    def delete(self):
        # Delete ticker from the list
        self.lTickers.takeItem(self.lTickers.currentRow())
    
    def delall(self):
        # Delete all tickers from the list
        self.lTickers.clear()

    def saveEvent(self):
        # Save the predictions to a CSV file
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "","CSV files (*.csv)", options=options)
        if fileName:
            with open(fileName, mode='w') as csv_file:
                fieldnames = ['symbol', 'x_split', 'prev. close', 'prediction', 'avg. err', 'accuracy', 'variation']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for ticker in self.predictions:
                    writer.writerow({'symbol':ticker, 
                                    'x_split':self.predictions[ticker]['x_split'],
                                    'prev. close':self.predictions[ticker]['prev. close'], 
                                    'prediction':self.predictions[ticker]['prediction'],
                                    'avg. err':self.predictions[ticker]['avg. err'],
                                    'accuracy':self.predictions[ticker]['accuracy'],
                                    'variation':self.predictions[ticker]['variation']})

    def getTickers(self):
        # Get the tickers from the list widget
        tickers = []
        nTickers = self.lTickers.count()
        for x in range(nTickers):
            tickers.append(self.lTickers.item(x).text())
        return tickers

    def validateTickers(self, tickers):
        # Validate tickers 
        validTickers = []

        # Read global tickers list
        allTickers = pd.read_csv('ticker_symbols.csv')
        # Drop unnecesary observations
        allTickers = allTickers[allTickers['Country']=='USA']
        # Drop unnecesary features
        allTickers = allTickers.drop('Country', axis=1)

        # Validate tickers
        for ticker in tickers:
            idx = allTickers[allTickers['Ticker']==ticker]
            if not(idx.empty):
                line = '{} \t| {} \t| {} \t| Valid'.format(idx.iloc[0]['Ticker'],
                                        idx.iloc[0]['Name'],
                                        idx.iloc[0]['Category Name'])
                # line = '{:6} \t| {:35} \t| {:30} \t| Valid'.format(idx.iloc[0]['Ticker'],
                #                     idx.iloc[0]['Name'],
                #                     idx.iloc[0]['Category Name'])
                validTickers.append(ticker)
            else:
                line = '<font style="color:Red;">{} \t| *** Invalid ***</font>'.format(ticker)
            self.tTickerDisplay.append(line)
        return validTickers

    def getStockData(self, ticker, start='2015'):
        # Get OHLC data for the stock
        df = web.DataReader(ticker, data_source='yahoo', start='2015')
        # Drop unnecesary features
        df = df[['Open', 'High', 'Low', 'Close']]
        # Shift the data one row
        df['open'] = df['Open'].shift(1)
        df['high'] = df['High'].shift(1)
        df['low'] = df['Low'].shift(1)
        df['close'] = df['Close'].shift(1)
        return df
    
    def predict(self):
        # Clear ticker details window
        self.tTickerDisplay.clear()
        # Cleaer the predictions table
        self.tPredictions.setRowCount(0)
        self.statusBar.showMessage('Initiating prediction...', 1000)

        # get the tickers list
        tickers = self.getTickers()

        # Validate tickers
        self.statusBar.showMessage('Validating Tickers...', 1000)
        tickers = self.validateTickers(tickers)
        self.statusBar.showMessage('Tickers validated...', 1000)

        self.predictions = {}

        # Create pre-processing pipeline
        # Create hyper-parameters
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

        steps = [('imputation', imp),
                ('scaler', StandardScaler()),
                ('lasso', Lasso())]
        pipeline = Pipeline(steps)

        parameters = {'lasso__alpha':np.arange(0.0001,10,.0001),
                    'lasso__max_iter':np.random.uniform(100,100000,4)}

        for index, ticker in enumerate(tickers):
            self.statusBar.showMessage('Analysing {} - Fetching OHLC data'.format(ticker), 1000)
            
            # Get ticker data
            self.start = self.cStartPeriod.itemText(self.cStartPeriod.currentIndex())
            dfTicker = self.getStockData(ticker, self.start)
            # Extract value from last close
            last = dfTicker[['Open', 'High', 'Low', 'Close']].iloc[-1].values
            # Eliminate last close from training set
            dfTicker = dfTicker[:-1]

            reg = rcv(pipeline, parameters, cv=5)

            # Split dependent and independent variables
            X = dfTicker[['open', 'high', 'low', 'close']]
            y = dfTicker['Close']

            avg_err = {}

            # Determine best train/test data split
            self.statusBar.showMessage('Analysing {} - Calculating optimal train/test split...'.format(ticker), 1000)

            for t in np.arange(50, 97, 3):
                split = int(t*len(X)/100)
                reg.fit(X[:split], y[:split])
                best_alpha = reg.best_params_['lasso__alpha']
                best_iter = reg.best_params_['lasso__max_iter']
                reg1 = Lasso(alpha=best_alpha, max_iter=best_iter)
                X = imp.fit_transform(X, y)
                reg1.fit(X[:split], y[:split])
                
                # Make the predictions
                dfTicker['P_C_%i'%t] = 0
                dfTicker.iloc[split:,dfTicker.columns.get_loc('P_C_%i'%t)] = reg1.predict(X[split:])
                
                # Calculate the mean absolute error (MAE) of the prediction model
                avg_err[t] = mean_absolute_error(dfTicker['Close'][split:],dfTicker['P_C_%i'%t][split:])
                
            # Recommed training/testing split for single prediction
            min_err = min(avg_err, key=lambda k: avg_err[k])

            self.statusBar.showMessage('Analysing {} - Prediction in progress...'.format(ticker), 1000)

            ########################################
            # Predict single value
            t=min_err
            split = int(t*len(X)/100)
            reg.fit(X[:split], y[:split])
            best_alpha = reg.best_params_['lasso__alpha']
            best_iter = reg.best_params_['lasso__max_iter']
            reg1 = Lasso(alpha=best_alpha, max_iter=best_iter)
            X = imp.fit_transform(X, y)
            reg1.fit(X[:split], y[:split])

            # Determine model accuracy
            self.statusBar.showMessage('Analysing {} - Calculating prediction accuracy....'.format(ticker), 1000)
            accuracies = cross_val_score(estimator=reg1, X=X, y=y, cv=10)

            # Predict new closing value
            new = reg1.predict([last])

            # Add prediction to dictionary
            self.predictions[ticker] = {'x_split':min_err, 
                    'prev. close':last[3], 
                    'prediction':new[0], 
                    'avg. err':avg_err[min_err],
                    'accuracy':accuracies.mean(),
                    'variation':accuracies.std()*2}

            #########################################

            self.statusBar.showMessage('Analysing {} - Prediction complete.'.format(ticker), 1000)
            
            # Update prediction progress bar
            self.progressBar.setValue((index+1)/len(tickers)*100)
        
        # Display the results
        # Configure table
        
        # Disable table editing
        self.tPredictions.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        tline = 0
        for pred in self.predictions:
            self.tPredictions.insertRow(tline)       # Insert a row in the table
            self.tPredictions.setItem(tline, 0, QtWidgets.QTableWidgetItem(pred))
            self.tPredictions.setItem(tline, 1, QtWidgets.QTableWidgetItem('{}'.format(self.predictions[pred]['x_split'])))
            self.tPredictions.setItem(tline, 2, QtWidgets.QTableWidgetItem('{:.2f}'.format(self.predictions[pred]['prev. close'])))
            self.tPredictions.setItem(tline, 3, QtWidgets.QTableWidgetItem('{:.2f}'.format(self.predictions[pred]['prediction'])))
            self.tPredictions.setItem(tline, 4, QtWidgets.QTableWidgetItem('{:.2f}'.format(self.predictions[pred]['avg. err'])))
            self.tPredictions.setItem(tline, 5, QtWidgets.QTableWidgetItem('{:.2f}'.format(self.predictions[pred]['accuracy'])))
            self.tPredictions.setItem(tline, 6, QtWidgets.QTableWidgetItem('{:.2f}'.format(self.predictions[pred]['variation'])))
            tline += 1


#############
# Main loop
#############
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())