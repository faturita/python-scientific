'''
Basic Tensorflow2 walkthrough and snippet cookbook.  
You can start here adding whatever you want to have a functional running code.

Run this code with ann3 environment.

Tensorflow is fast, and allows automatic differentiation.


# OpemMP sometimes raises coredumps, try export KMP_DUPLICATE_LIB_OK=TRUE

Sources: 
* Deep Learning with Tensorflow 2 and Keras, Antonio Gulli et al, 2019
* Intro to Tensorflow and Deep Learning, Dr. Michael Fairbank

'''
# %%
# Basic Tensorflow model.
import tensorflow as tf 

W = tf.Variable( tf.ones(shape=(2,2)), name="W")
b = tf.Variable( tf.zeros( shape=(2)), name="b")

@tf.function
def model(x):
    return W * x + b

# TF has lazy evaluation.  This means that the code is not executed until you finally need it.
out_a = model([1,0])

print( out_a )

# %%
a=tf.constant([[5,6],[8,9]])
b=tf.constant([[1,2],[3,4]])
c=tf.multiply(a,b)                      # Hadamart product
print(c)

# %%
a=tf.constant([[5,6],[8,9]])
b=tf.constant([[1,2],[3,4]])
c=tf.matmul(a,b)                   
print(c)

# %%
a=tf.constant([[5,6],[8,9]])
b=tf.constant([[1,2],[3,4]])
c=tf.greater(a,b)             
print(c)


# %%
a=tf.constant([[1,2],[3,-4]], tf.float32)
c=tf.cast(a, tf.int32)                 
print(c)

# %%
a=tf.constant(2, tf.float32)
c=tf.add(a,a)                
print(c)

# %%
a=tf.constant([2,3,5,3,3], tf.float32)
c=tf.reduce_max(a)
d=tf.argmax(a)              
print(c)
print(d)

# %%
# Automatic differentiation, wonder of tensorflow
x=tf.Variable(5.0, tf.float32)
with tf.GradientTape() as g:
        #g.watch(x)
        y=tf.multiply(x,x)
        dydx=g.gradient(y,[x])             
print(c)
#print(dydx.numpy())

# %%
# Automatic differentiation, wonder of tensorflow
x=tf.Variable(4.0, tf.float32)
y=tf.Variable(2.0, tf.float32)
with tf.GradientTape(persistent=True) as g:
        f=tf.pow(x,tf.constant(2.0, tf.float32))*tf.constant(3.0, tf.float32)+y
        dydx=g.gradient(f,[x,y])             
print(f)
print(dydx[0].numpy(), dydx[0].numpy())

# %%
print("Let's move forward with the classic MNIST prediction.")

accuracies = []


import numpy as np
from tensorflow import keras

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
