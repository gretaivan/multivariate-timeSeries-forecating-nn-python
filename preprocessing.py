# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite

This module contains time-series data pre-processing functions
"""

import numpy as np

#data transformation using logarithm to reduce trend
def log_transform(df):
    df_log = np.log(df)
    #plt.plot(ts_log)
    return df_log