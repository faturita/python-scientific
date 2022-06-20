'''
==================
OnePassClassifier
==================

. 
                         _ 
 _._ _..._ .-',     _.._(`)) 
'-. `     '  /-._.-'    ',/ 
   )         \            '. 
  / _    _    |             \ 
 |  a    a    /              | 
 \   .-.                     ;   
  '-('' ).-'       ,'       ; 
     '-;           |      .' 
        \           \    / 
        | 7  .__  _.-\   \ 
        | |  |  ``/  /`  / 
       /,_|  |   /,_/   / 
          /,_/      '`-' 


Show how small tweaks on the architecture help to achieve better performance.

Environment: advkeras

'''
# %%
print("Let's move forward with the classic MNIST prediction.")

accuracies = []


import numpy as np
from tensorflow import keras
import tensorflow as tf

epochs = 30
batch_size = 128
verbose = 1
nb_classes = 10
n_hidden = 128
validation_split = 0.2

mnist = keras.datasets.mnist
( X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 60k rows, 28x28
reshaped = 784

X_train = X_train.reshape( 60000, reshaped)
X_test  = X_test.reshape(10000, reshaped)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, nb_classes)

# %%
# Sequential model
model = tf.keras.models.Sequential()
model.add( keras.layers.Dense(nb_classes, 
        input_shape=(reshaped,),
        name='dense_layer',
        activation='softmax'))

# Objective function
# MSE, binary_crossentropy, categorical_crossentropy, and so on
model.compile(  optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_split=validation_split)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)
accuracies.append( test_acc )

# %%
# Change the network architecture and rebuild the model.

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(n_hidden,
        input_shape=(reshaped,),
        name='dense_layer',
        activation='relu'))
model.add(keras.layers.Dense(n_hidden,
        name='dense_layer_2',
        activation='relu'))
model.add(keras.layers.Dense(nb_classes,
        name='dense_layer_3',
        activation='softmax'))

model.summary()

model.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(X_train, Y_train,
            batch_size = batch_size, 
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)  
accuracies.append( test_acc )         

# %%
# Let's use dropout to improve performance
dropout = 0.3

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(n_hidden,
            input_shape=(reshaped,),
            name='dense_layer',
            activation='relu'))
model.add(keras.layers.Dropout(dropout))
model.add(keras.layers.Dense(n_hidden,
        name='dense_layer_2',
        activation='relu'))
model.add(keras.layers.Dense(nb_classes,
        name='dense_layer_3',
        activation='softmax'))

model.summary()

model.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(X_train, Y_train,
            batch_size = batch_size, 
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)   
accuracies.append( test_acc )

print(accuracies)

# %%
