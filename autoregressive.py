"""
========================
Autoregressive Modelling
========================

Basic Autoregressive Modelling

https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/

"""
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
signals = pd.read_csv('data/blinking.dat', delimiter=' ', usecols=[2], names = ['timestamp','counter','eeg','attention','meditation','blinking'])

from matplotlib import pyplot
from pandas.plotting import lag_plot
lag_plot(signals)
pyplot.show()


from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot

values = DataFrame(signals.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)

from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error

autocorrelation_plot(signals)
pyplot.show()

# create lagged dataset
values = DataFrame(signals.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

print("Train set (%d,%d)" % train.shape)
print("Test set (%d,%d)" % test.shape)
 
# persistence model
def model_persistence(x):
	return x
 
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)

test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()


from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

# split dataset
X = signals.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

