# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:05:02 2020

@author: ladyg

https://www.kaggle.com/gouherdanishiitkgp/eda-and-forecasting-brent-oil-prices
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#packages for plotting candlestick graph
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc

#packages for the prediction of time-series data
from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


print(os.listdir("input"))

#read brent oil prices from kaggle
#Usability10.0
#License
#U.S. Government Works
#commodities and futures trading
#https://www.kaggle.com/mabusalah/brent-oil-prices/data#

brent = pd.read_csv("input/BrentOil.csv")

brent.head()

#DATA PREPROCESSING
#date conversion to the standard format


print(type(brent['Date']))
print(brent)
brent['Date'] = pd.to_datetime(brent['Date'], format="%d-%b-%y")
print(brent)

#DATA EXPLORATION 
#visualisation
import seaborn as sns
from matplotlib import pyplot as plt
#plot the price change
brentViz = sns.lineplot(x='Date',y='Price',data = brent)
plt.title("Brent Oil Price Trend")

#plot for specific period
def plot_price_trend(df, start_date, end_date):
     mask = (brent['Date'] > start_date) & (brent['Date'] <= end_date)
     sBrent = brent.loc[mask]
     plt.figure(figsize = (10,5))
     chart = sns.lineplot(x='Date', y='Price', data = sBrent)
     plt.title("Brent Oil Price Trend")
    
plot_price_trend(brent, '2017-01-01','2019-01-01')



#FORECAST MODELS

#Step 1. PROPHET

from fbprophet import Prophet
p = Prophet()
#https://towardsdatascience.com/a-quick-start-of-time-series-forecasting-with-a-practical-example-using-fb-prophet-31c4447a2274

#Step 2. transform data
#prophet requires date column to be called ds and the sample y
pro_brent = brent
pro_brent.columns = ['ds','y']
pro_brent.head()


#step 3. fitting
#fit data into the model for 90 days forecast
p.fit(pro_brent)
future = p.make_future_dataframe(periods = 90)
forecast = p.predict(future)

"""
Step 4. check the forecast components: trend, weekly
yearly seasonality. Each component has lower and upper 
confidence interval data
"""

forecast.head()

p.plot_components(forecast)

p.plot(forecast)

#comparison of predicted price against real, y column for real prices
cmp_brent = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']].join(pro_brent.set_index('ds'))
cmp_brent.head()

#need to leave out some test data
cmp_brent.tail(5)

"""
Step 5. Comparison and viisualisation of the data
Comnining predicted data alongside original data
"""

#visualisation
plt.figure(figsize=(17,8))
#plt.plot(cmp_df['yhat_lower'])
#plt.plot(cmp_df['yhat_upper'])
plt.plot(cmp_brent['yhat'])
plt.plot(cmp_brent['y'])
plt.legend()
plt.show()

def plot_price_forecast(brent,start_date, end_date):
    """
    This function filters the dataframe for the specified date range and 
    plots the actual and forecast data.
    
    Assumption: 
    - The dataframe has to be indexed on a Datetime column
    This makes the filtering very easy in pandas using df.loc
    """
    cmp_brent = brent.loc[start_date:end_date]
    plt.figure(figsize=(17,8))
    plt.plot(cmp_brent['yhat'])
    plt.plot(cmp_brent['y'])
    plt.legend()
    plt.show()
    
plot_price_forecast(cmp_brent,'2017-01-01','2020-01-01')


#ARIMA
from statsmodels.tsa.arima_model import ARIMA    # ARIMA Modeling
from statsmodels.tsa.stattools import adfuller   # Augmented Dickey-Fuller Test for Checking Stationary
from statsmodels.tsa.stattools import acf, pacf  # Finding ARIMA parameters using Autocorrelation
from statsmodels.tsa.seasonal import seasonal_decompose # Decompose the ARIMA Forecast model

#Step 1. pre-processing/data transfomation
#arima model requires date column to be set as index, transform data
arima_brent = brent.set_index('ds')
arima_brent.head()

#Step 2. pre-pricessing : data stationarity
# funtion that plots rolling statistics and checks stationarity, using 
#Dickey fuller test for the given data set
def test_stationarity(ts):
    
    #rolling statistics
    #rolling mean 
    mean = ts.rolling(window=12).mean()
    #standard deviation
    std = ts.rolling(window=12).std()

    #Plot rolling statistics:
    originalData = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(mean, color='red', label='Rolling Mean')
    stdDeviation = plt.plot(std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Dickey-Fuller test:
    print('Dickey-Fuller Test:')
    dftest = adfuller(ts['y'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    

#call stationarity funtion for brent data set
test_stationarity(arima_brent)

"""
Observation - The null hypothesis of ADF test is the 
Time series is NOT stationary. We see that the Test 
Statistic (-1.98) is higher than 10% Critical Value 
(-2.56). This means this result is statistically 
significant at 90% confidence interval and so, we 
fail to reject the null hypothesis.

This means that our time series data is NOT stationary.


#reference: 
#https://www.kaggle.com/freespirit08/time-series-for-beginners-with-arima
"""

#Step 3. Data correlation identification
"""
Some definitions -
Correlation - Describes how much two variables depend on each other.
Partial Correlation - When multiple variables are involved, two variables may have direct relation as well as indirect relation (i.e x1 and x3 are related and x2 and x3 are related. Due to this indirect relation, x1 and x2 might be related). This is called partial correlation.
Auto Correlation - In a time series data, variable at a time step is dependent upon its lag values. This is called auto-correlation (i.e. variable depending upon its own values)
Partial Autocorrelation - describes correlation of a variable with its lag values after removing the effect of indirect correlation.
"""

#import partion autocorrelation library
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(arima_brent)
plot_pacf(arima_brent)

# Implementing own function to create ACF plot
def get_acf_plot(ts):
    #calling acf function from stattools
    y = ts['y']
    lag_acf = acf(y, nlags=500)
    plt.figure(figsize=(16, 7))
    plt.plot(lag_acf, marker="o")
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    
def get_pacf_plot(ts):
    #calling pacf function from stattools
    y = ts['y']
    lag_pacf = pacf(y, nlags=50)
    plt.figure(figsize=(16, 7))
    plt.plot(lag_pacf, marker="o")
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')

get_acf_plot(arima_brent)
get_pacf_plot(arima_brent)


#Step 4. Make data stationary
#Log transformation
ts_log = np.log(arima_brent)
plt.plot(ts_log)

#moving average of last 12 values
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color
         ='red')

# Finding the difference between log values and moving average
ts_diff = ts_log - moving_avg
ts_diff.head(12)

"""
No diferrence for 11 days only on 12th
"""
ts_diff.dropna(inplace=True)
ts_diff
test_stationarity(ts_diff)

# Exponentially weighted moving average 
expwighted_avg = ts_log.ewm(halflife=12).mean()

plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

#Step 5. 
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

#Seasonal decomposition var and plot
from statsmodels.tsa.seasonal import seasonal_decompose
ts_log.head()
decomposition = seasonal_decompose(ts_log, period = 30)

trend = decomposition.trend
trend
seasonal = decomposition.seasonal
seasonal
residual = decomposition.resid
residual

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#fittinf cleared data to the test
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
ts_log_decompose.head()
ts_log_decompose.tail()
ts_log_decompose.columns = ['ds','y']

#the decomposed data does not suit the created f-tion
#therefore transformation needed
ts_log_decompose_trnf = ts_log_decompose.to_frame()
ts_log_decompose_trnf.columns = ['y']
ts_log_decompose_trnf

test_stationarity(ts_log_decompose_trnf)

"""
Step 6. ARIMA Models
"""
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=0)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

