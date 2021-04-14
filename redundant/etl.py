# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite

ETL - Export, Transform, Load
Simulating ETL program to provide the 
univariate and multivariate data
"""

import numpy as np # linear algebra
import pandas as pd # data processing from csv
import matplotlib.pyplot as plt
import seaborn as sns
import os

#packages for plotting candlestick graph
import matplotlib.ticker as mticker

#packages for the prediction of time-series data
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error 
import matplotlib.dates as mdates

class ETL:
    
    usd, brent, dax, nasdaq, nasdaq100, wti, xau = ''
    
    def _init_(self):
        #univariate = getUniVar()
        #multivariate = getMultiVar()  
          
        #usd/eur currency data - target data
        self.usd = pd.read_csv("input/usd.csv")
        #usdRaw['xTime'] = pd.to_datetime(usdRaw.xTime , format = '%d/%m/%Y')          
        #brent oil data
        self.brent = pd.read_csv("input/BrentOil.csv", parse_dates=['Date'])
        #print(brent)
        #frankfurt index data
        self.dax =  pd.read_csv("input/dax.csv")
        #print(dax)
        #nasdaq data
        self.nasdaq =  pd.read_csv("input/nasdaq.csv")
        #nasdaq 100 data
        self.nasdaq100 =  pd.read_csv("input/nasdaq100.csv")
        #crude oil data
        self.wti =  pd.read_csv("input/wti.csv")
        #gold data
        self.xau =  pd.read_csv("input/xau.csv")
    
        #target data time filter dates
        self.start = '1999-04-01'
        self.end ='2020-02-25'
        
        
    """
    Loads all csv data available from input file
    """
    
                
        
    def getDataSet(self):      
        
        """
        DATA REFORMATING and PREPARATION
        """
        #filter
        def selectByDate(df, start_date, end_date):
           filter = (df['Date'] >= start_date) & (df['Date'] <= end_date)
           df = df.loc[filter]
           return df
             
                
        def indexDf(df):
            dfDate = df.Date
            df = df.drop(['Date'], axis=1)
            df.index =dfDate
            return df
                
        #returns formated data:
        #sorts, assign to date , resample and replace nan with 0, filters by date
        def formatToSeries(x, start_date, end_date):
            #replace nan with 0 
            x = x.fillna(0)
            x.columns = ['Date', 'y']
            x['Date'] = pd.to_datetime(x.Date , format = '%d/%m/%Y') 
            x = x.sort_values(by=['Date'], ascending=[True])
                
            #xDate = x.Date
            #x = x.drop(['Date'], axis=1)
            #x.index = indexDf(x)
            x = indexDf(x)
            
                
            #when resampled and missed days added they values are equal to day before
            x = x.resample('D').ffill().reset_index() 
            x = selectByDate(x,start_date, end_date)
            x = x.sort_values(by=['Date'], ascending=[True])
            x = indexDf(x)
            return x
            
        """
        DATA CLEANING
        further handling for the missing values
        null values replaced by 0 must become median value
        """
        
        def removeMissingValues(data):
            columList = data.columns
            #print(columList)
                
            for col in columList:
                for i in range(0, len(data)):
                    try:
                        j = i + 1 #look up next row
                        #pd.Series(data[col][i].value == 0)
                        if(data[col][i] == 0):
                            #find the next row that contains the value
                            while (data[col][j] == 0):
                                j = j + 1
                                #print("j incremented to: ", j)   
                                #replace with average of values before and after  
                                #print("current j: ", j)
                                
                            data[col][i] = (data[col][i-1] + data[col][j])/2   
                            #print(data[col][i-1], " and ",  data[col][j])
                            #print("the row: ", i, "value now is: ", data[col][i] )
                    except IndexError: 
                        print("The index at: ", i, " went out of bounds but error has been handled")
                        
                    print("The missing values replacement has been successful")
                    return data
        
        #combine separate data structures to multivariate data
        def mergeToMultiVar(df,df1):
            multiVar = pd.merge(df, df1, on="Date")
            return multiVar
        
        #methods for data preparation
        def prepareUniVar():
            uniVar = formatToSeries(self.usd, self.start, self.end)
            uniVar = removeMissingValues(self.usd)
            return uniVar
            
        def prepareData():
            
            #execute formating function on all data sets
            brent = formatToSeries(self.brent, self.start, self.end)
            usd = formatToSeries(self.usd, self.start, self.end)
            dax = formatToSeries(self.dax, self.start, self.end)
            nasdaq = formatToSeries(self.nasdaq, self.start, self.end)
            nasdaq100 = formatToSeries(self.nasdaq100, self.start, self.end)
            wti = formatToSeries(self.wti, self.start, self.end)
            xau = formatToSeries(self.xau ,self.start, self.end)
            data_list = [self.usd, self.brent, self.dax, self.nasdaq, self.nasdaq100, self.wti, self.xau]
        
            #replace 0 to median value
            brent = removeMissingValues(brent)
            usd = removeMissingValues(usd)
            dax = removeMissingValues(dax)
            nasdaq = removeMissingValues(nasdaq)
            nasdaq100 = removeMissingValues(nasdaq100)
            wti = removeMissingValues(wti)
            xau = removeMissingValues(xau) 
                
            i = 0
        
            for asset in data_list:
                    
                if(i == 0):
                    multiVar = mergeToMultiVar(data_list[i], data_list[i+1])
                else:
                    multiVar = mergeToMultiVar(multiVar, data_list[i+1])
                i = i+1
                print(i)
                if(i == 6):
                    break
        
            multiVar.columns = ['y_usd','y_brent','y_dax','y_nasdaq','y_nasdaq100','y_wti','y_xau']
        
        
        print("Do you want to extract univariate or multivariate data?")
        print("1 - for univariate")
        print("2 - for multivariate")
        type = input()
        
        
        if(type == 1):
            #def getUnivariate():
            univariate = prepareUniVar()
            return univariate      
            
        elif(type == 2):     
            #def getMultiVar():
            multivariate = prepareData()
            return multivariate
        
        else:
            print("There are only two types of data sets")
    
    
        
        
    
