# -*- coding: utf-8 -*-
'''
Collection of functions for statistical analysis of time series.
Commonly used for stock analysis

Created on Tue Mar 19 22:36:34 2019
'''

__version__ = '1.0'
__author__ = 'Stein Castillo'

# Imports
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import seaborn as sns

# Define file management functions
def create_csv(symbol, start='', end=''):
    ''' Download stock data and save resutls in a csv file 

    Usage: create_csv('symbol', start_date, end_date)

    Parameters:
        symbol: <string>: Valid stock ticker
        start_date: <string>: 'yyyy-mm-dd'
        end_date: <string>: 'yyyy-mm-dd'

    Returns: saves a CSV file with the symbol information
    '''
    # Check parameters
    if symbol== '' or start == '' or end=='':
        raise Exception ('[ERROR: create_csv] Invalid parameters!')
    
    # Download stock information from the web
    tmpdf = web.DataReader(symbol, 'yahoo', start, end)
    # Create CSV file
    tmpdf.to_csv('{}.csv'.format(symbol))
    
def get_csv(symbols=['SPY'], start='', end=''):
    ''' 
    Read the CSV file with the source data into a dataframe

    Usage: df = get_csv([symbols], start_date, end_date)

    Parameters:
        symbol: [list]: Valid stock tickers. i.e. ['AAPL', 'PM', 'DIS']
        start_date: <string>: 'yyyy-mm-dd'
        end_date: <string>: 'yyyy-mm-dd'

    Returns: Dataframe with the closing values for the selected symbols
    '''
        
    # Validate dates
    if start == '' or end=='':
        raise Exception ('[ERROR: get_csv] Invalid dates!')

    # Create a pandas date index
    dates = pd.date_range(start, end)
    # Create dataframe with dates index
    df = pd.DataFrame(index=dates)
    
    # Read the rest of symbols
    for symbol in symbols:

        # Validate file exists    
        file_check = Path('{}.csv'.format(symbol))
        if not(file_check.is_file()):
            # file does not exist
            raise Exception('[ERROR: get_csv] File {}.csv does not exist!'.format(symbol))
           
        dftemp = pd.read_csv('{}.csv'.format(symbol),
                         index_col = 'Date',
                         parse_dates = True,
                         usecols = ['Date', 'Adj Close'],
                         na_values = ['nan'])
        # Rename column to prevent duplication
        dftemp = dftemp.rename(columns={'Adj Close':symbol.upper()})
        df = df.join(dftemp)   # Use default how='left'
    df = df.dropna()
    return df

# Define Statistical analysis functions
def sma(df, window=2):
    ''' 
    Calculate simple moving average (SMA)

    Usage: smadf = sma(df, window)

    Parameters:
        df : [dataframe] Dataframe with the stock daily closing values
        window: [interger]: Number of periods to use for first sma calculation
                default value = 2

    Returns: dataframe with the simple moving averages by stock per day
    '''
    return df.rolling(window).mean()

def rolling_standard_deviation(df, window=2):
    '''
    Calculate rolling standard deviation 

    Usage: stdf = rolling_standard_deviation(df, window)
     Parameters:
        df : [dataframe] Dataframe with the stock daily closing values
        window: [interger]: Number of periods to use for first sma calculation
                default value = 2
    '''
    return df.rolling(window).std()

def daily_returns(df):
    ''' 
    Returns daily returns calculation:
        daily returns[t] = price[t] / price[t-1] -1 
        
    Usage: drdf = daily_returns(df)
    Parameters:
        df: [dataframe] Dataframe with the stock daily closing values
    '''
    daily_return = (df / df.shift(1)) - 1
    daily_return = daily_return.fillna(0)
    return daily_return

def daily_returns_v2(df):
    ''' 
    Returns daily returns calculation
    
    Usage: drdf = daily_returns(df)
    Parameters:
        df: [dataframe] Dataframe with the stock daily closing values
    '''
    daily_return = df.pct_change()
    return daily_return

def rolling_cum_returns(df):   
    ''' 
    Calculate rolling cummulative returns 
    '''
    symbol = df.columns.values[0]
    rret = df.values
    rret = (rret[-1] / rret - 1)*100
    # Convert the results to dataframe
    rret = pd.DataFrame(rret)
    # Reverse the order of the dataframe
    rret = rret.iloc[::-1]
    # Set the date index to the dataframe
    rret.set_index(df.index.values, inplace=True)
    rret = rret.rename(columns={0:symbol})
    return rret

def get_bollinger_bands(sma, window=2):
    ''' 
    Calculate upper and lower bollinger bands

    Usage: upper_band, lowerband = get_bollinger_bands(smadf, window)

    Parameters:
        smadf : [dataframe] Dataframe with the stock daily simple moving average value
        window: [interger]: Number of periods to use for first sma calculation
             default value = 2 
    '''
    upper_band = sma + sma.rolling(window).std()*2
    lower_band = sma - sma.rolling(window).std()*2
    return upper_band, lower_band

def normalize(df):
    ''' 
    Normalize dataframe values 
    '''
    return df / df.iloc[0]

# Define portfolio value functions
def portfolio_daily_value(df, allocation, start_value):
    '''
    Calculate the daily value of a portfolio
    '''
    # Step 2: Normalize the prices
    df_norm = normalize(df)
    # Step 3: Multiply normalized values * allocation
    df_alloced = df_norm * allocation
    # Step 4: Multiply allocated values * start value
    df_pos_val = df_alloced * start_value
    # Step 5: Calculate the value for each day
    df_port_val = df_pos_val.sum(axis=1).rename('value')
    df_port_val = pd.DataFrame(df_port_val)
    return df_port_val

def sharpe_ratio(rp, rf=0, samples=252):
    k = np.sqrt(samples)
    a = np.mean(rp - rf)
    b = np.std(rp)
    sr = k * (a / b)
    return sr

# Define plotting functions
def scatter_plot(dfx, dfy):
    ''' 
    Create a scatter plot and calculate alpha and beta 
    '''
    # Get the series title
    xlabel = dfx.name
    ylabel = dfy.name
    # Plot scatter chart
    ax = sns.regplot(x=dfx, y=dfy)
    ax.set(title='{} vs. {}'.format(xlabel, ylabel))
    # Calculate alpha and beta
    beta, alpha = np.polyfit(dfx, dfy, 1)

    # Draw vertical and horizontal line
    ax.axvline(0, color='g', linestyle=':')
    ax.axhline(0, color='g', linestyle=':')
    
    # Annotate the chart with alpha and beta values
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.text(xmax-(xmax*.40), ymin-(ymin*.20), r'$\beta$: {:.4f}'.format(beta))
    ax.text(xmax-(xmax*.40), ymin-(ymin*.10), r'$\alpha$: {:.4f}'.format(alpha))
    
def histogram_plot(dfdr, bins=10):
    ''' 
    Plot a histogram with mean and std variation lines 
    '''
    # get series title
    label = dfdr.columns[0]
    # Calculate mean and standard variation
    dfmean = dfdr.mean()
    dfmean = dfmean[0]
    dfstd = dfdr.std()
    dfstd = dfstd[0]
    
    # Note that this chart includes a density curve
    ax = sns.distplot(dfdr, bins=bins)
    ax.set(title='{} (bins={})'.format(label, bins))
    
    # Plot mean vertical line
    ax.axvline(dfmean, color='g', linestyle='dashed', linewidth=2)
    
    # Plot standard deviation boundaries
    ax.axvline(dfstd, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(-dfstd, color='r', linestyle='dashed', linewidth=2)
    
    # Calculate kurtosis and annotate the chart
    # Calculate kurtosis
    kurdf = dfdr.kurtosis()
    kurdf = kurdf[0]
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.text(xmax-(xmax*.40), ymax-(ymax*.10), 'Kurtosis: {:.4f}'.format(kurdf))
    
def hist_plot(dfdr, bins=10):
    dfdr.hist(bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    plt.plot(x, mlab.normpdf(x, mean, std_dev), 'r')
    plt.show()

def var_parm(drdf, confidence=0.99):
    ''' 
    Returns Value at risk (VaR) using de variance-covariance approach
    based on the level of confidence provided as input parameter
    
    Usage: var_parm(drdf, confidence)
    Example: var_parm(drpm['PM'], 0.99)
    Parameters:
        drdf: [DataFrame column]. Dataframe column with the daily returns
        confidence: [float]: value between 0 and 0.99 indicating the level
        of confidence that will be used to calculate the Var. Note that this
        value must be < 1.00
    '''
    if confidence > 0.99:
        return 0
    dfmean = np.mean(drdf)
    dfstd_dev = np.std(drdf)
    var = norm.ppf(1-confidence, dfmean, dfstd_dev)
    return var
    
