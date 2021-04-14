# -*- coding: utf-8 -*-
"""

@author: Greta Ivanauskaite

ETL simulation for file system
"""


# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite
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
    
#specify the needed date range    
start = '1999-04-01'
end ='2020-02-25'


def changeDates(startDate, endDate):
    start = startDate
    end = endDate

"""
LOADING THE DATA
"""  

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



#execute formating function on all data sets
brent = formatToSeries(brent, start, end)

usd = formatToSeries(usdRaw, start, end)
dax = formatToSeries(dax, start, end)
nasdaq = formatToSeries(nasdaq, start, end)
nasdaq100 = formatToSeries(nasdaq100, start, end)
wti = formatToSeries(wti, start, end)
xau = formatToSeries(xau ,start, end)


data_list = [usd, brent, dax, nasdaq, nasdaq100, wti, xau]

"""
DATA CLEANING

#further handling for the missing values
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
    #print("The missing values replacement has been successful")
    return data

brent = removeMissingValues(brent)
usd = removeMissingValues(usd)
dax = removeMissingValues(dax)
nasdaq = removeMissingValues(nasdaq)
nasdaq100 = removeMissingValues(nasdaq100)
wti = removeMissingValues(wti)
xau = removeMissingValues(xau) 


#combine separate data structures to multivariate data
def mergeToMultiVar(df,df1):
    multiVar = pd.merge(df, df1, on="Date")
    return multiVar
    
i = 0

for asset in data_list:
    
    if(i == 0):
        multiVar = mergeToMultiVar(data_list[i], data_list[i+1])
    else:
        multiVar = mergeToMultiVar(multiVar, data_list[i+1])
    i = i+1
    #print(i)
    if(i == 6):
        break

multiVar.columns = ['y_usd','y_brent','y_dax','y_nasdaq','y_nasdaq100','y_wti','y_xau']

print("Data has been loaded")
del start
del end
del asset
del i
del usdRaw
print("Redundant values has been discarded")
return