# -*- coding: utf-8 -*-
"""

@author: Greta Ivanauskaite 
Evaluation of the predicted values agains actual
"""
#https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/
#very good!
#https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score


def statistics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    evs = explained_variance_score(actual, predicted)
    me = max_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print('MSE: ', mse,'\nRMSE:',rmse,'\nMAE: ',mae,'\nEVS: ',evs, '\nME:  ',me,'\nR2:  ',r2)


