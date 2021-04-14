# -*- coding: utf-8 -*-
"""
@author: Greta Ivanauskaite w1670486
"""
# transform time series to supervised learning format
from numpy import array
from pandas import DataFrame
from math import sqrt
from numpy import array
from numpy import mean
from numpy import std
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

# split a univariate data into sequence samples, where output depends on n_steps

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values


def split_seq(sequence, n_steps):
    
	x, y = list(), list()
    
	for i in range(len(sequence)):
		# find the end by adding time steps
		end = i + n_steps
        
		# check if not out of range
		if end > len(sequence)-1:
			break
        
		#input and output
		seq_x, seq_y = sequence[i:end], sequence[end]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)

def toSupervised(data, n_input, n_out):
# flatten data
	#data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# split a multivariate data array to sequence samples, where output depends on n_steps
def split_multiSeq(sequences, n_steps):
    
	x, y = list(), list()
    
	for i in range(len(sequences)):
		end = i + n_steps
		
		if end > len(sequences):
			break
        
		#input and output
		seq_x, seq_y = sequences[i:end, :-1], sequences[end-1, -1]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)

# transform input from [samples, features] to [samples, timesteps, features]
def to3D(doubleArray): 
    X = doubleArray.reshape(doubleArray.shape[0], doubleArray.shape[1], 1)
    #print(X.shape)
    return X

def to3Df(doubleArray, features): 
    X = doubleArray.reshape(doubleArray.shape[0], doubleArray.shape[1], features)
    return X

def split2D(doubleArray, ratio):
    ratio = len(doubleArray) * ratio
    ratio = int(ratio)
    print("the data is split at sample: ", ratio)
    trainX, testX = doubleArray[:ratio, :], doubleArray[ratio:, :]
    return trainX, testX    

def splitArray(array, ratio):
    ratio = len(array) * ratio
    ratio = int(ratio)
    print("the data is split at sample: ", ratio)
    trainY, testY = array[:ratio], array[ratio:]
    return trainY, testY

#reshape to the dataset with specified dimentions
def dimReshape(seq, dimension):
    seq = seq.reshape((len(seq), dimension))
    return seq

# split a univariate for multistep
def split_seq_for_multistep(sequence, n_steps_in, n_steps_out):
	x, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end = i + n_steps_in
		out_end = end + n_steps_out
		# check if we are beyond the sequence
		if out_end > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end], sequence[end:out_end]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)

# split a multivariate for multistep
def split_multiSeq_for_multistep(sequences, n_steps_in, n_steps_out):
	x, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end = i + n_steps_in
		out_end = end + n_steps_out-1
		# check if we are beyond the dataset
		if out_end > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end, :-1], sequences[end-1:out_end, -1]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)

