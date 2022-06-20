'''

Based on https://cnvrg.io/keras-custom-loss-functions/
https://keras.io/guides/writing_your_own_callbacks/

'''


# %%
print("Measuring Performance on Keras models")

accuracies = []


import numpy as np
from tensorflow import keras
import tensorflow as tf 

# Metrics 
## Accuracy
## Precision
## Recall
## F-1 Score
## MSE: Mean square error: L2 loss, 1/n Sum (^Yi - Y)^2
## MAE: Mean absolute error, L1 error, Sum | ^Yi - Y | / n
## Confussion Matrix
## Logarithmic Loss
## ROC
## Cross Entropy
## Binary Cross Entropy
## Categorical Cross Entropy, multi class classification
## Hinge Loss (SVM), l = max(0,1 - y . Y)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(1,), activation='relu'),
    keras.layers.Dense(1)
])

# Training set.
x_train = np.asarray( [[10.0],[20.0], [30.0],[40.0],[50.0],[60.0],[10.0],[20.0]])
y_train = np.asarray( [6,12,18,24,30,36,6,12])


# Let's compile the model with MSE as loss function
model.compile(loss='mse', optimizer='Adam')
model.fit(x_train, y_train, epochs=1000)

print( f'Output Prediction for {x_train[0]}({y_train[0]}):{model.predict(x_train[0])}')

# Now, let's see if the final prediction improves using MAE.
model.compile(loss='mae', optimizer='Adam')
model.fit(x_train, y_train, epochs=1000)

print( f'Output Prediction for {x_train[0]}({y_train[0]}):{model.predict(x_train[0])}')


# Now let's use a new stop condition on the network.  This will check if the 'loss' value is less than 0.1
class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        if (logs.get('loss')<0.1):

            print("Loss bellow threshold ! ---- Stopping !")
            self.model.stop_training = True

    
callback = new_callback()

model.compile(loss='mae', optimizer='Adam')
model.fit(x_train, y_train, epochs=100000, callbacks=[callback])

print( f'Output Prediction for {x_train[0]}({y_train[0]}):{model.predict(x_train[0])}')


# Now let's see if we can use a customized loss function, and a customized callback to verify when to stop.
from tensorflow.python.ops import math_ops
from keras import backend as K

def customLoss( y_true, y_pred):
  diff = math_ops.squared_difference(tf.cast(y_pred, tf.float32), tf.cast(y_pred, tf.float32))  #squared difference
  loss = K.mean(diff, axis=-1) #mean
  loss = loss / 10.0
  return loss

class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}): 
        if (logs.get('loss')<0.1):

            print("Loss bellow threshold ! ---- Stopping !")
            self.model.stop_training = True
callback = new_callback()

model.compile(loss=customLoss, optimizer='Adam')
model.fit(x_train, y_train, epochs=1000, callbacks=[callback])

print( f'Output Prediction for {x_train[0]}({y_train[0]}):{model.predict(x_train[0])}')