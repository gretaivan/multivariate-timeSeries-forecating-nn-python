# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite
Module for data analysis and exploration
Contains functions for data and model visualisation
"""
from statsmodels.tsa.stattools import adfuller
import pandas as pd # data processing from csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from keras.models import load_model
from statsmodels.tsa.stattools import adfuller

"""
def plot_dates_values(data):
    dates = data["timestamp"].to_list()
    values = data["value"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates, values)
    plt.show()
   """ 
    
#Perform Dickey-Fuller test:
def adf_test(df): 
    print ('Dickey-Fuller Test:')
    test = adfuller(df, autolag='t-stat', regression='ct')
    output = pd.Series(test[0:4], index=['Test Statistic','p-value','Used Lag','Number of Observations Used for ADF'])
    for i,value in test[4].items():
       output['Critical Value (%s)'%i] = value
    output['BIC predicted lag value '] = test[5]   
    print (output)
    
    

#Rolling mean and standard devation visualisation
def plot_rollingStatistics(df):
    #Determing rolling statistics window 
    mean = df.rolling(30).mean()
    #print(mean)
    #standard deviation
    std = df.rolling(30).std()
    #Plot rolling statistics:
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(mean, color='red', label='Rolling Mean')
    std = plt.plot(std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    

def plotModelLearning(history,title):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

def plotPrediction(actual, predicted, title):
    plt.figure(figsize=(8, 4))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_tick_params(rotation=45)
    plt.plot(actual, label='actual', color='blue', alpha=0.8, linewidth=2)
    plt.plot(predicted, label='predicted', color='red', linewidth=1)
    plt.title(title)
    plt.legend(loc='best')
    plt.show(block=False)

       