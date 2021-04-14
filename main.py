# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite
Implementation of the data prediction methods and model for 3 time steps 
and only with necessary data pre-processing

"""

"""
DATA LOADING
"""

import etl as etl
import pandas as pd # data processing from csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import dataAnalysis as da
import transformData as td
#import keras as k
#from tensorflow.keras.models import Sequential
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import utils
#from keras.callbacks import ModelCheckpoint
#rom keras.models import load_model

print(etl.os.listdir("input"))
    
#usd/eur currency data - target data
usd = pd.read_csv("input/usd.csv")
    
#brent oil data
brent = pd.read_csv("input/BrentOil.csv", parse_dates=['Date'])

#frankfurt index data
dax =  pd.read_csv("input/dax.csv")

#nasdaq data
nasdaq =  pd.read_csv("input/nasdaq.csv")

#nasdaq 100 data
nasdaq100 =  pd.read_csv("input/nasdaq100.csv")

#crude oil data
wti =  pd.read_csv("input/wti.csv")

#gold data
xau =  pd.read_csv("input/xau.csv")

#print(usd.head())
usd.describe()
#print(best.head())
brent.describe()
dax.describe()
nasdaq.describe()
nasdaq100.describe()
wti.describe()
xau.describe()

usd.head(10)
brent.head(10)
dax.head(10)
#Date range for the data
start = '1999-04-01'
end ='2020-02-25'

"""
raw data formating to series and inserting missing daily values
"""
#execute formating function on all data sets to convert to TIMESERIES dataframe
#format raw data, add missing days
brent = etl.formatToSeries(brent, start, end)
usd = etl.formatToSeries(usd, start, end)
dax = etl.formatToSeries(dax, start, end)
nasdaq = etl.formatToSeries(nasdaq, start, end)
nasdaq100 = etl.formatToSeries(nasdaq100, start, end)
wti = etl.formatToSeries(wti, start, end)
xau = etl.formatToSeries(xau, start, end) 

#replace mising values
brent = etl.removeMissingValues(brent)
usd = etl.removeMissingValues(usd)
dax = etl.removeMissingValues(dax)
nasdaq = etl.removeMissingValues(nasdaq)
nasdaq100 = etl.removeMissingValues(nasdaq100)
wti = etl.removeMissingValues(wti)
xau = etl.removeMissingValues(xau) 


# separate data sets combination to multivariate data set

#print(usd.head())
usd.describe()
#print(best.head())
brent.describe()
dax.describe()
nasdaq.describe()
nasdaq100.describe()
wti.describe()
xau.describe()

"""
DATA EXPLORATION
"""

fig, axs = plt.subplots(7, figsize=(12, 8), sharex=True, constrained_layout=True)#, gridspec_kw={'hspace': 0}
#fig.text(0.5, 0.04, 'Assets overview', ha='center', va='center')
#fig.suptitle('Assets overview')
axs[0].plot(usd)
axs[0].set_title('USD/EUR')
axs[1].plot(brent)
axs[1].set_title('Brent crude Oil')
axs[2].plot(dax)
axs[2].set_title('Frankfurt index')
axs[3].plot(nasdaq)
axs[3].set_title('NASDAQ')
axs[4].plot(nasdaq100)
axs[4].set_title('NASDAQ100')
axs[5].plot(wti)
axs[5].set_title('WTI Crude Oil')
axs[6].plot(xau)
axs[6].set_title('Gold')

usd.description()

brent.head()
dax.head()
nasdaq.head()
nasdaq100.head()
wti.head()
xau.head()


#DATA STATIONARITY TEST
import dataAnalysis as da
  
#test = adfuller(usd, autolag='t-stat', regression='ct')    
#print(test)
print('USD')
da.adf_test(usd)
print('BRENT')
da.adf_test(brent)   
print('DAX')
da.adf_test(dax) 
print('NASDAQ')
da.adf_test(nasdaq) 
print('NASDAQ 100')
da.adf_test(nasdaq100) 
print('WTI')
da.adf_test(wti) 
print('XAU')
da.adf_test(xau) 

"""
#decompose into Trend, Seasonality and Residual
import statsmodels.api as sm

def decomposePlot(df):
    sm.tsa.seasonal_decompose(df).plot()
    result = sm.tsa.stattools.adfuller(df)
    plt.show()
    
decomposePlot(usd)
decomposePlot(brent)
"""

#visualise rolling statistics and standard deviation
da.plot_rollingStatistics(usd)
da.plot_rollingStatistics(brent)
da.plot_rollingStatistics(dax)
da.plot_rollingStatistics(nasdaq)
da.plot_rollingStatistics(nasdaq100)
da.plot_rollingStatistics(wti)
da.plot_rollingStatistics(xau)

import preprocessing as pp

usd_log = pp.log_transform(usd)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(usd_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(usd_log, label='Original')
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

usd_log_decompose = residual
usd_log_decompose.dropna(inplace=True)
test_stationarity(usd_log_decompose)

da.adf_test(usd_log_decompose)


brent_log = pp.log_transform(brent)

fig, axs = plt.subplots(2, figsize=(12, 8), sharex=True, constrained_layout=True)#, gridspec_kw={'hspace': 0}
#fig.text(0.5, 0.04, 'Assets overview', ha='center', va='center')
#fig.suptitle('Assets overview')
axs[0].plot(usd_log)
axs[0].set_title('USD/EUR')
axs[1].plot(brent_log)
axs[1].set_title('Brent crude Oil')
axs[2].plot(dax)
axs[2].set_title('Frankfurt index')
axs[3].plot(nasdaq)
axs[3].set_title('NASDAQ')
axs[4].plot(nasdaq100)
axs[4].set_title('NASDAQ100')
axs[5].plot(wti)
axs[5].set_title('WTI Crude Oil')
axs[6].plot(xau)
axs[6].set_title('Gold')


"""
Data split to training and testing values
90% of data us used for training and 10% for test
"""
#univariate data
trainUV = usd[:int(0.9*(len(usd)))]
testUV = usd[int(0.9*(len(usd))):]

trainUV.head()    
testUV.tail()

"""
VAR MODEL
"""
data_list = [usd, brent, dax, nasdaq, nasdaq100, wti, xau]
#Input preparation
i = 0

for asset in data_list:
    
    if(i == 0):
        multiVardf = etl.mergeToMultiVar(data_list[i], data_list[i+1])
    else:
        multiVardf = etl.mergeToMultiVar(multiVardf, data_list[i+1])
    i = i+1
    #print(i)
    if(i == 6):
        break
    
multiVardf.columns = ['y_usd','y_brent','y_dax','y_nasdaq','y_nasdaq100','y_wti','y_xau']
multiVardf_data = multiVardf[['y_usd','y_brent','y_dax','y_nasdaq','y_nasdaq100','y_wti','y_xau']]

multiVardf_data.index = pd.DatetimeIndex(multiVardf.index)

#logarithmic transformation
multiVardf_data_log = np.log(multiVardf_data).diff().dropna()



#discard uneeded values
del asset
del i
print("Redundant values has been discarded")

#training and test data for multivariate data
train_multiVardf = multiVardf_data[:int(0.9*(len(multiVardf_data)))]
test_multiVardf = multiVardf_data[int(0.9*(len(multiVardf_data))):]

print(train_multiVardf.shape, test_multiVardf.shape)

#fit VAR model
from statsmodels.tsa.vector_ar.var_model import VAR

model_VAR = VAR(endog=train_multiVardf)
model_fit = model_VAR.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(test_multiVardf))

cols = multiVardf_data.columns
#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)), columns=[cols])
#pred = pd.DataFrame(index=range(0,len(prediction)), columns=range(7))
#pred = test
#pred = pred.fillna(0)
prediction[0][0]

for j in range(0, 7):
    for i in range(0, len(prediction)):
       #print("j: ",j, " i: ", i)
       pred.iloc[i][j] = prediction[i][j]     
       
#dateRange = test.index
#pred = pred.drop(['Index'], axis=1)
#pred.index = dateRange
#multiVar.plot(x="usd", y="brent", kind="scatter", figsize=(10,10));

#check rmse
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#print(type(pred['y_usd']))
#rmse = sqrt(mean_squared_error(pred[i], test[i]))
#print(rmse)
#test.columns
#pred.columns = [test.columns]
#testPred = pd.DataFrame(prediction)
#testPred.columns = cols
#pred['y_usd']
#prediction
actual = test_multiVardf['y_usd']      
"""
for i in cols:
    print(i)
    predicted = testPred[i]
    actual = test[i]
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(predicted, actual)))
"""
#make final predictions
#model_VAR = VAR(endog=train_multiVardf)
#model_fit = model_VAR.fit()
#yhat = model_fit.forecast(model_fit.y, steps=764)
#print(yhat.shape)

date = test_multiVardf.index
predictionDated = pd.DataFrame(data=prediction,index=date)
predictionDated.columns = multiVardf.columns
predictionDated.plot()

da.plotPrediction(actual, predictionDated['y_usd'], 'VAR model performance without pre-processing')

predictionDated['y_usd'].to_excel('Model3 VAR no pre-processing.xlsx')    


"""
DEEP LEARNING
"""

"""
data transformation to 2d arrays for univaraite prediction ch6.
"""
#import transformData as td

#prepare univariate data as sequence sample for MLP 3 timesteps
steps = 3
uniVar = pd.DataFrame.to_numpy(usd['y'])
print(uniVar.shape)
uniVarX, uniVarY = td.split_seq(uniVar, steps)     
print(uniVarX.shape, uniVarY.shape)

# show 5 samples
for i in range(5):
	print(uniVarX[i], uniVarY[i])
      
# split into train and test
ratio = 0.9

len(uniVarX)

uniVarX_train, uniVarX_test = td.split2D(uniVarX, ratio)
print(uXtrain.shape)

uniVarY_train, uniVarY_test = td.splitArray(uniVarY, ratio)


"""
Univariate MLP
ID: 1
"""


# MLP model

"""
model = keras.Sequential([
    #keras.Input(shape=(784))
    
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
"""

#one input and one output tensors
model = Sequential(name="Univariate MLP model for input with 3 lagged values")
#add layers
#activate as rectified linear f-tion, see time steps as separate feature
model.add(Dense(100, activation='relu', input_dim=steps))
model.add(Dense(1))

#compilation with Adam stochastic gradient descent and optimise by mse - loss f-tion
model.compile(optimizer='adam', loss='mse')

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
#enable automatic best model saving for reuse
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# fit model
#model.fit(uniVarX, uniVarY, epochs=2000, verbose=0, validation_split=0.9)
#history = model.fit(uXtrain, uYtrain, validation_data=(uXtest, uYtest), epochs=4000, verbose=0)
#history = model.fit(uXtrain, uYtrain, validation_data=(uXtest, uYtest), epochs=4000, verbose=0, callbacks=[es, mc])
#history_model = model.fit(uniVarX_train, uniVarY_train, validation_split=0.1, epochs=4000, verbose=0, callbacks=[es])
history_model = model.fit(uniVarX_train, uniVarY_train, validation_split=0.33, epochs=4000, batch_size=32, verbose=0, callbacks=[es])

# evaluate the model
#_, train_eval = model.evaluate(uXtrain, uYtrain, verbose=0)
#_, test_eval = model.evaluate(uXtest, uYtest, verbose=0)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(uniVarX_test, uniVarY_test)
print("test loss, test acc:", results)
# load the saved model
#saved_model = load_model('best_model.h5')
#saved_model.summary()
model.summary()
model.summary()
#evaluate the model
#_, train_eval = saved_model(uXtrain, uYtrain, verbose=0)
#_, test_eval = saved_model(uXtest, uYtest, verbose=0)
#_,train_eval = model(uniVarX_train, uniVarY_train, verbose=0)
#_,test_eval = model(uniVarX_test, uniVarY_test, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_eval, test_eval))

print("Number of weights after calling the model:", len(model.weights))  # 6


print(history_modelB.history.keys())
#from matplotlib import plt
# plot train"ing hist"ory
"""
plt.plot(history_model.history['loss'], label='train')
plt.plot(history_model.history['val_loss'], label='test')
plt.title('Model learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
"""
plt.plot(history_modelB.history['loss'], label='loss')
plt.plot(history_modelB.history['val_loss'], label='value loss')
plt.title('Loss / Mean Squared Error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


"""
Use model to predict
"""
#demonstrate prediction
#x_input = uXtest
#x_input = x_input.reshape((1, steps))
model_prediction = model.predict(uniVarX_test, verbose=0)
#print(model_prediction)
#actualX,actualY = td.splitArray(uniVar, ratio)

#date = usd['Date']
datePredictionX, datePredictionY = td.splitArray(date, ratio)

print(model_prediction.shape, datePredictionY.shape)

yhat_model = model_prediction[:,0]


yhat_model = pd.DataFrame(data=yhat_model, index=datePredictionY)
actual = pd.DataFrame(data=uniVarY_test, index=datePredictionY)


plt.figure(figsize=(8, 4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation=45)
plt.plot(actual, label='actual usd', color='blue', alpha=0.8, linewidth=2)
plt.plot(yhat_model, label='predicted usd', color='red', linewidth=1)
plt.title('USD prediction with MLP and univariate data')
plt.legend(loc='best')
plt.show(block=False)

yhat_model.to_excel('MLP 1.xlsx')



"""
Another univariate MLP model id:2
"""

model_uniVarAlt = Sequential(name="Univariate MLP2 with 3 time steps")
model_uniVarAlt.add(Dense(500, input_dim=steps, activation='relu'))
model_uniVarAlt.add(Dense(1))
model_uniVarAlt.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])



n_train = 6869 - 686
uniVarX_train, x_validation = uniVarX_train[:n_train, :], uniVarX_train[n_train:, :]
uniVarY_train, y_validation = uniVarY_train[:n_train], uniVarY_train[n_train:]

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
history_modelAlt = model_uniVarAlt.fit(uniVarX_train, uniVarY_train, validation_data=(x_validation, y_validation), epochs=8000,  callbacks=[es])#verbose=,


_,train_acc = model_uniVarAlt.evaluate(uniVarX_train, uniVarY_train)
_,test_acc = model_uniVarAlt.evaluate(x_validation, y_validation)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# list all data in history
print(history_mv.history.keys())

"""
# plot loss during training
plt.subplot(212)
plt.title('Loss')
plt.plot(history_modelAlt.history['loss'], label='train')
plt.plot(history_modelAlt.history['val_loss'], label='test')
plt.legend()
# plot mse during training
plt.subplot(222)
plt.title('Mean Squared Error')
plt.plot(history_modelAlt.history['mean_squared_error'], label='train')
plt.plot(history_modelAlt.history['val_mean_squared_error'], label='test')
plt.legend()
plt.show()
"""
da.plotModelLearning(history_modelAlt, 'Loss - MAE')

"""
Prediction
"""
#demonstrate prediction

prediction_model2 = model.predict(uniVarX_test)

#double check the date index
print(prediction_model2.shape, datePredictionY.shape)

yhat_model2 = prediction_model2[:,0]
yhat_model2 = pd.DataFrame(data=yhat_model, index=datePredictionY)

print(actual.shape, yhat_model2.shape)

da.plotPrediction(actual, yhat_model2, 'Univariate MLP 2')

yhat_model2.to_excel('MLP 2.xlsx')


for i in yhat_model:
    diff = actual - yhat_model
    
print(diff.head(30))


actual.to_excel('actual.xlsx')    
diff.to_excel('actual.xlsx')   
    

"""
MLP for Multivariate data
Multivariate data conversion to supervided problem
"""
steps = 3
#at the moment the data is vertical
usd_seq = pd.DataFrame.to_numpy(usd['y'])
brent_seq = pd.DataFrame.to_numpy(brent['y'])
dax_seq = pd.DataFrame.to_numpy(dax['y'])
nasdaq_seq = pd.DataFrame.to_numpy(nasdaq['y'])
nasdaq100_seq = pd.DataFrame.to_numpy(nasdaq100['y'])
xau_seq = pd.DataFrame.to_numpy(xau['y'])
wti_seq = pd.DataFrame.to_numpy(wti['y'])

#reshape sequences to single dataset
usd_seq = td.dimReshape(usd_seq, 1)
brent_seq = td.dimReshape(brent_seq, 1)
dax_seq = td.dimReshape(dax_seq, 1)
nasdaq_seq = td.dimReshape(nasdaq_seq, 1)
nasdaq100_seq = td.dimReshape(nasdaq100_seq, 1)
xau_seq = td.dimReshape(xau_seq, 1)
wti_seq = td.dimReshape(wti_seq, 1)

from numpy import hstack
#reshape data, where row represents time step and column is feature
#doubling usd value because the last one is to represent expected  output
multiVar = hstack((usd_seq, brent_seq, dax_seq, nasdaq_seq, nasdaq100_seq, xau_seq, wti_seq, usd_seq))
print(multiVar.shape)
#convert into input/output according the lag
multiVarX, multiVarY = td.split_multiSeq(multiVar, steps)

print(multiVarX.shape,multiVarY.shape)

#get number of inputs, must be 21 as 7 features x 3steps
n_input = multiVarX.shape[1] * multiVarX.shape[2]

#convert the input to the vectors, where vector length = features x time steps
multiVarX = multiVarX.reshape((multiVarX.shape[0], n_input))

print(multiVarX.shape)
print(multiVarY.shape)

# split into train and test
ratio = 0.9

multiVarX_train, multiVarX_test = td.split2D(multiVarX, ratio)
print(multiVarX_train.shape)

multiVarY_train, multiVarY_test = td.splitArray(multiVarY, ratio)
"""
Build MLP model for multivariate
ID: 4
"""
# define MLP model for multivar
model_mv = Sequential()
model_mv.add(Dense(100, activation='relu', input_dim=n_input))
model_mv.add(Dense(1))
model_mv.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])

#validation data 10%
n_train = 6869 - 686
multiVarX_train, multiX_validation = multiVarX_train[:n_train, :], multiVarX_train[n_train:, :]
multiVarY_train, multiY_validation = multiVarY_train[:n_train], multiVarY_train[n_train:]

# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2000)

history_mv4 = model_mv.fit(multiVarX_train, multiVarY_train, validation_data=(multiX_validation,multiY_validation), epochs=4000, callbacks=[es])
# demonstrate prediction

model_mv.summary()
model_mv.get_config()
#from keras.utils.vis_utils import plot_model
#plot_model(model_mv, to_file='model4.png')

# list all data in history
print(history_mv4.history.keys())

# plot history
# summarize history for loss
plt.figure(figsize=(15, 10))
plt.plot(history_mv4.history['loss'])
plt.plot(history_mv4.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


"""
Use model for prediction
"""
mutiVar_prediction = model_mv.predict(multiVarX_test, verbose=0)
#print(mutiVar_prediction)
    
#demonstrate prediction
date = usd.index
datePredictionX, datePredictionY = td.splitArray(date, ratio)

model4_yhat = mutiVar_prediction[:,0]

model4_yhat = pd.DataFrame(data=model4_yhat, index=datePredictionY)
actual = pd.DataFrame(data=multiVarY_test, index=datePredictionY)

da.plotPrediction(actual, model4_yhat, 'Model 4: Multivariate MLP prediction performance')

plt.figure(figsize=(8, 3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation=45)
plt.plot(yhat, label='prediction')
plt.plot(actual, label='actual')
plt.title('USD prediction with MLP and multivariate data')
plt.legend(loc='best')
plt.show(block=False)

"""
Multistep univariate MLP: 5 days
ID: 5
"""
n_steps, n_steps_out = 3, 15

uniVar = pd.DataFrame.to_numpy(usd['y'])
print(uniVar.shape)
uniVarX_multistep, uniVarY_multistep = td.split_seq_for_multistep(uniVar, n_steps, n_steps_out)

print(uniVarX_multistep.shape, uniVarY_multistep.shape)

ratio = 0.9
uniVarX_multistep_train, uniVarX_multistep_test = td.split2D(uniVarX_multistep, ratio)
print(uniVarX_multistep_train.shape)

uniVarY_multistep_train, uniVarY_multistep_test = td.split2D(uniVarY_multistep, ratio)


#validation data 10%
n_train = 6866 - 686
uniVarX_multistep_train, uniVarX_multistep_v= uniVarX_multistep_train[:n_train, :], uniVarX_multistep_train[n_train:, :]
uniVarY_multistep_train, uniVarY_multistep_v = uniVarY_multistep_train[:n_train, :], uniVarY_multistep_train[n_train:, :]

# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
model5 = Sequential()
model5.add(Dense(100, activation='relu', input_dim=n_steps))
model5.add(Dense(n_steps_out))
model5.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
# fit model
history_m5 = model5.fit(uniVarX_multistep_train, uniVarY_multistep_train, validation_data=(uniVarX_multistep_v,uniVarY_multistep_v), epochs=2500, callbacks=[es])

da.plotModelLearning(history_m5, 'Model 5')

# demonstrate prediction
m5_prediction = model5.predict(uniVarX_multistep_test, verbose=0)
half_test, expected = uniVarX_multistep_test[:749,:],uniVarY_multistep_test[13:764,:]
m5_prediction2 =  model5.predict(half_test, verbose=0)
print(m5_prediction.shape, uniVarY_multistep_test.shape)
print(m5_prediction2.shape)


datePredictionY = datePredictionY[2:764]
model5_yhat = pd.DataFrame(data=m5_prediction, index=datePredictionY)
actual_multi = pd.DataFrame(data=uniVarY_multistep_test, index=datePredictionY)
da.plotPrediction(actual_multi[0], model5_yhat[14],'Model 5:  15 days prediction')
da.plotPrediction(actual_multi[10][700:762], actual_multi[10][700:762],'Model 5:  15 days prediction')



date = usd.index
datePredictionX, datePredictionY = td.splitArray(date, ratio)
datePredictionY.shape
actual_multi = pd.DataFrame(data=expected, index=datePredictionY[15:764])
# demonstrate prediction
model5_yhat = pd.DataFrame(data=m5_prediction2, index=datePredictionY[15:764])
da.plotPrediction(actual_multi[0], model5_yhat[0],'Model 5:  15 days prediction')
# define model

"""
Multistep univariate MLP: 5 days
ID: 6
"""
n_steps, n_steps_out = 3, 15


usd_seq = pd.DataFrame.to_numpy(usd['y'])
brent_seq = pd.DataFrame.to_numpy(brent['y'])
dax_seq = pd.DataFrame.to_numpy(dax['y'])
nasdaq_seq = pd.DataFrame.to_numpy(nasdaq['y'])
nasdaq100_seq = pd.DataFrame.to_numpy(nasdaq100['y'])
xau_seq = pd.DataFrame.to_numpy(xau['y'])
wti_seq = pd.DataFrame.to_numpy(wti['y'])

#reshape sequences to single dataset
usd_seq = td.dimReshape(usd_seq, 1)
brent_seq = td.dimReshape(brent_seq, 1)
dax_seq = td.dimReshape(dax_seq, 1)
nasdaq_seq = td.dimReshape(nasdaq_seq, 1)
nasdaq100_seq = td.dimReshape(nasdaq100_seq, 1)
xau_seq = td.dimReshape(xau_seq, 1)
wti_seq = td.dimReshape(wti_seq, 1)

from numpy import hstack
multiVar = hstack((usd_seq, brent_seq, dax_seq, nasdaq_seq, nasdaq100_seq, xau_seq, wti_seq, usd_seq))
print(multiVar.shape)



#split to general input output
multiVarX_multi, multiVarY_multi = td.split_multiSeq_for_multistep(multiVar, n_steps, n_steps_out)
print(multiVarX_multi.shape)
#get number of inputs, must be 21 as 7 features x 3steps
n_input = multiVarX_multi.shape[1] * multiVarX_multi.shape[2]
#n_output = multiVarY_multi.shape[1] * multiVarY_multi.shape[2]
#convert the input to the vectors, where vector length = features x time steps
multiVarX_multi = multiVarX_multi.reshape((multiVarX_multi.shape[0], n_input))
#multiVarY_multi = multiVarY_multi.reshape((multiVarY_multi.shape[0], n_output))
#training and test data
ratio = 0.9
multiVarX_multistep_train, multiVarX_multistep_test = td.split2D(multiVarX_multi, ratio)
print(multiVarX_multistep_train.shape)

multiVarY_multistep_train, multiVarY_multistep_test = td.split2D(multiVarY_multi, ratio)

#validation data 10%
n_train = 6866 - 686
multiVarX_multistep_train, multiVarX_multistep_v= multiVarX_multistep_train[:n_train, :], multiVarX_multistep_train[n_train:, :]
multiVarY_multistep_train, multiVarY_multistep_v = multiVarY_multistep_train[:n_train, :], multiVarY_multistep_train[n_train:, :]

# fit model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
model6 = Sequential()
model6.add(Dense(100, activation='relu', input_dim=n_input))
model6.add(Dense(n_steps_out))
model6.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
# fit model
history_m6 = model6.fit(multiVarX_multistep_train, multiVarY_multistep_train, validation_data=(multiVarX_multistep_v,multiVarY_multistep_v), epochs=2500, callbacks=[es])

da.plotModelLearning(history_m6, 'Model 6')

# demonstrate prediction
#m6_prediction = model6.predict(uniVarX_multistep_test, verbose=0)
m6_test, expected = multiVarX_multistep_test[:757,:],multiVarY_multistep_test[:762,:]
m6_prediction =  model6.predict(m6_test, verbose=0)
print(m6_prediction.shape, expected.shape)



datePredictionY = datePredictionY[2:764]
model5_yhat = pd.DataFrame(data=m5_prediction, index=datePredictionY)
actual_multi = pd.DataFrame(data=uniVarY_multistep_test, index=datePredictionY)
da.plotPrediction(actual_multi[0], model5_yhat[14],'Model 5:  15 days prediction')
da.plotPrediction(actual_multi[10][700:762], actual_multi[10][700:762],'Model 5:  15 days prediction')



date = usd.index
datePredictionX, datePredictionY = td.splitArray(date, ratio)
datePredictionY.shape
actual_multi = pd.DataFrame(data=expected, index=datePredictionY[:762])
# demonstrate prediction
model6_yhat = pd.DataFrame(data=m6_prediction, index=datePredictionY[7:764])
da.plotPrediction(actual_multi[0], model6_yhat[0],'Model 6:  15 days prediction')


"""
DATA PRE-PROCESSING
"""
#univariate values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_usd= scaler.fit_transform(usd)
scaled_multiVar = scaler.fit_transform(multiVardf)

print(scaled_usd)


scaled_usd_df = pd.DataFrame(data=scaled_usd, index=usd.index, columns=['usd'])
scaled_multiVar_df = pd.DataFrame(data=scaled_multiVar, index=usd.index, columns=['usd', 'brent', 'dax', 'nasdaq', ' nasdaq100', 'wti', 'xau'])

import dataFormater as d

n_input = 3
usd_x, usd_y = td.toSupervised(scaled_usd, n_input, 3)
usd_x = usd_x.reshape((usd_x.shape[0], n_input))

ratio = 0.9
usd_x_train, usd_x_test = td.split2D(usd_x,ratio)
usd_y_train , usd_y_test = td.split2D(usd_y,ratio)

timesteps = usd_x_train.shape[0]
features = usd_x_train.shape[0]

#multivariate

#training and test data for multivariate data
scaled_multiVar_train = scaled_multiVar_df[:int(0.9*(len(scaled_multiVar_df)))]
scaled_multiVar_test = scaled_multiVar_df[int(0.9*(len(scaled_multiVar_df))):]



"""
Models
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#Univariate LTSM id: 7
model7 = Sequential()
model7.add(LSTM(200, activation='relu', ndim=3))#input_shape=(3, 1)
model7.add(Dense(100, activation='relu'))
model7.add(Dense(1))
model7.compile(loss='mse', optimizer='adam')
# fit network
model7.fit(usd_x_train, usd_y_train, epochs=100)
	
usd_x_test = usd_x.reshape((usd_x.shape[0], n_input))
model7.predict(usd_x_test)

#fit VAR model i:8
from statsmodels.tsa.vector_ar.var_model import VAR

model_VAR = VAR(endog=scaled_multiVar_train)
model_fit = model_VAR.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(scaled_multiVar_test))

cols = scaled_multiVar_df.columns

pred = pd.DataFrame(index=range(0,len(prediction)), columns=[cols])

prediction[0][0]

for j in range(0, 7):
    for i in range(0, len(prediction)):
       #print("j: ",j, " i: ", i)
       pred.iloc[i][j] = prediction[i][j]     
       
actual = scaled_multiVar_test['usd'] 

date = actual.index
predictionDated = pd.DataFrame(data=prediction,index=date)
predictionDated.columns = scaled_multiVar_test.columns
predictionDated.plot()

da.plotPrediction(actual, predictionDated['usd'], 'VAR model performance')

predictionDated['y_usd'].to_excel('Model3 VAR no pre-processing.xlsx')    


"""
DEEP LEARNING
"""

"""
data transformation to 2d arrays for univaraite prediction ch6.
"""
#import transformData as td

#prepare univariate data as sequence sample for MLP 3 timesteps
steps = 3
uniVar = pd.DataFrame.to_numpy(scaled_usd_df)
print(uniVar.shape)
uniVarX, uniVarY = td.split_seq(uniVar, 28)     
print(uniVarX.shape, uniVarY.shape)

train = td.series_to_supervised(uniVar, 28)
# show 5 samples
for i in range(5):
	print(uniVarX[i], uniVarY[i])
      
# split into train and test
ratio = 0.9
train = usd_x.reshape((train[0], 28))
len(uniVarX)

uniVarX_train, uniVarX_test = td.split2D(train, ratio)
print(uXtrain.shape)

uniVarY_train, uniVarY_test = td.splitArray(uniVar[28:], ratio)

steps = 28
uniVar = pd.DataFrame.to_numpy(scaled_usd_df)
print(uniVar.shape)
uniVarX, uniVarY = td.split_seq(uniVar, steps)     
print(uniVarX.shape, uniVarY.shape)

# show 5 samples
for i in range(5):
	print(uniVarX[i], uniVarY[i])
      
# split into train and test
ratio = 0.9

uniVarX_train, uniVarX_test = td.split2D(uniVarX, ratio)
print(uXtrain.shape)

uniVarY_train, uniVarY_test = td.splitArray(uniVarY, ratio)


"""
Univariate MLP
ID: 9
"""
"""
usd_x_train, usd_x_test = td.split2D(usd_x,ratio)
usd_y_train , usd_y_test = td.split2D(usd_y,ratio)
"""
#one input and one output tensors
model9 = Sequential(name="model9")
#add layers
#activate as rectified linear f-tion, see time steps as separate feature
model9.add(Dense(100, activation='relu', input_dim=29))
model9.add(Dense(200, activation='relu'))
model9.add(Dense(1))

#compilation with Adam stochastic gradient descent and optimise by mse - loss f-tion
model9.compile(optimizer='adam', loss='mse')

# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
#enable automatic best model saving for reuse
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# fit model
#model.fit(uniVarX, uniVarY, epochs=2000, verbose=0, validation_split=0.9)
#history = model.fit(uXtrain, uYtrain, validation_data=(uXtest, uYtest), epochs=4000, verbose=0)
#history = model.fit(uXtrain, uYtrain, validation_data=(uXtest, uYtest), epochs=4000, verbose=0, callbacks=[es, mc])
#history_model = model.fit(uniVarX_train, uniVarY_train, validation_split=0.1, epochs=4000, verbose=0, callbacks=[es])
history_m9 = model9.fit(uniVarX_train, uniVarY_train, validation_split=0.1, epochs=500, verbose=0)

# evaluate the model
#_, train_eval = model.evaluate(uXtrain, uYtrain, verbose=0)
#_, test_eval = model.evaluate(uXtest, uYtest, verbose=0)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(uniVarX_test, uniVarY_test)
print("test loss, test acc:", results)
# load the saved model
#saved_model = load_model('best_model.h5')
#saved_model.summary()
model.summary()
model.summary()
#evaluate the model
#_, train_eval = saved_model(uXtrain, uYtrain, verbose=0)
#_, test_eval = saved_model(uXtest, uYtest, verbose=0)
#_,train_eval = model(uniVarX_train, uniVarY_train, verbose=0)
#_,test_eval = model(uniVarX_test, uniVarY_test, verbose=0)
#print('Train: %.3f, Test: %.3f' % (train_eval, test_eval))

print("Number of weights after calling the model:", len(model.weights))  # 6


print(history_modelB.history.keys())
#from matplotlib import plt
# plot train"ing hist"ory
"""
plt.plot(history_model.history['loss'], label='train')
plt.plot(history_model.history['val_loss'], label='test')
plt.title('Model learning curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
"""
plt.plot(history_modelB.history['loss'], label='loss')
plt.plot(history_modelB.history['val_loss'], label='value loss')
plt.title('Loss / Mean Squared Error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


"""
Use model to predict
"""
#demonstrate prediction
#x_input = uXtest
#x_input = x_input.reshape((1, steps))
model_prediction = model.predict(uniVarX_test, verbose=0)
#print(model_prediction)
#actualX,actualY = td.splitArray(uniVar, ratio)

#date = usd['Date']
datePredictionX, datePredictionY = td.splitArray(date, ratio)

print(model_prediction.shape, datePredictionY.shape)

yhat_model = model_prediction[:,0]


yhat_model = pd.DataFrame(data=yhat_model, index=datePredictionY)
actual = pd.DataFrame(data=uniVarY_test, index=datePredictionY)


plt.figure(figsize=(8, 4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_tick_params(rotation=45)
plt.plot(actual, label='actual usd', color='blue', alpha=0.8, linewidth=2)
plt.plot(yhat_model, label='predicted usd', color='red', linewidth=1)
plt.title('USD prediction with MLP and univariate data')
plt.legend(loc='best')
plt.show(block=False)

yhat_model.to_excel('MLP 1.xlsx')


"""
EVALUATION
"""
import evaluation as e
#MLP
e.statistics(actual, yhat_model)
#MLP
e.statistics(actual, yhat_model2)
#VAR
e.statistics(actual, predictionDated['y_usd'])
#Multivariate MLP 4
e.statistics(actual, model4_yhat)
#Model 5
e.statistics(actual_multi[0], model5_yhat[0])
#model 8
e.statistics(actual, predictionDated['usd'])
