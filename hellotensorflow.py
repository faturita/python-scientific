'''
Basic Tensorflow2 walkthrough and snippet cookbook.  
You can start here adding whatever you want to have a functional running code.

Run this code with ann3 environment.

Tensorflow is fast, and allows automatic differentiation.


# OpemMP sometimes raises coredumps, try export KMP_DUPLICATE_LIB_OK=TRUE

Sources: 
* Deep Learning with Tensorflow 2 and Keras, Antonio Gulli et al, 2019
* Intro to Tensorflow and Deep Learning, Dr. Michael Fairbank
* Chapter 3, Chollet Deep Learning book.

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
# Random Tensors
x = tf.random.normal( shape=(3,1), mean=0.,stddev=1.)
print(x)

x = tf.random.uniform( shape=(3,1), minval=0., maxval=1.)

# %%
# Assign values to Exisiting tf variables or subsets

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
v.assign(tf.ones((3, 1)))

v[0, 0].assign(3.)

# %%
# Automatic differentiation, wonder of tensorflow
x=tf.Variable(5.0, tf.float32)
with tf.GradientTape() as g:
    #g.watch(x)
    y=tf.multiply(x,x)              # x^2 is the function
    dydx=g.gradient(y,[x])          # So the derivative is 2*x

print(dydx)                         # You will see a 10 here.

# %%
# Automatic differentiation, wonder of tensorflow
x=tf.Variable(4.0, tf.float32)
y=tf.Variable(2.0, tf.float32)
with tf.GradientTape(persistent=True) as g:
    f=tf.pow(x,tf.constant(2.0, tf.float32))*tf.constant(3.0, tf.float32)+y
    dydx=g.gradient(f,[x,y])             
print(f)
print(dydx[0].numpy(), dydx[0].numpy())


#%%

def linear_layer(x):
    return 3*x + 2

@tf.function 
def simple_nn(x):
    return tf.nn.relu(linear_layer(x))

def simple_function(x):
    return 3*x

print(tf.autograph.to_code(simple_nn.python_function, experimental_optional_features=None))


# %%
import numpy as np
import matplotlib.pyplot as plt

# Define model and Loss

class Model(object):
    def __init__(self):
        self.W = tf.Variable(10.0)
        self.b = tf.Variable(-5.0)

    def __call__(self, inputs):
        return self.W * inputs + self.b

def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

model = Model()

# Define True weight and bias

TRUE_W = 3.0
TRUE_b = 2.0

# Obtain training data, Let's synthesize the training data with some noise.

NUM_EXAMPLES = 1000
inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Before we train the model let's visualize where the model stands right now.
# We'll plot the model's predictions in red and the training data in blue.

def plot(epoch):
    plt.scatter(inputs, outputs, c='b')
    plt.scatter(inputs, model(inputs), c='r')
    plt.title("epoch %2d, loss = %s" %(epoch, str(compute_loss(outputs, model(inputs)).numpy())))
    plt.legend()
    plt.draw()
    plt.ion()   # replacing plt.show()
    plt.pause(1)
    plt.close()

# Define a training loop
learning_rate = 0.1
for epoch in range(30):
    with tf.GradientTape() as tape:
        loss = compute_loss(outputs, model(inputs))

    dW, db = tape.gradient(loss, [model.W, model.b])

    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

    print("=> epoch %2d: w_true= %.2f, w_pred= %.2f; b_true= %.2f, b_pred= %.2f, loss= %.2f" %(
          epoch+1, TRUE_W, model.W.numpy(), TRUE_b, model.b.numpy(), loss.numpy()))
    plot(epoch + 1)


# Logistic Regression


# Parameters
learning_rate = 0.001
training_epochs = 6
batch_size = 600

# Import MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.reshape(x_train, [-1, 784]), y_train))
    .batch(batch_size)
    .shuffle(1000)
)

train_dataset = (
    train_dataset.map(lambda x, y:
                      (tf.divide(tf.cast(x, tf.float32), 255.0),
                       tf.reshape(tf.one_hot(y, 10), (-1, 10))))
)


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
model = lambda x: tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
# Minimize error using cross entropy
compute_loss = lambda true, pred: tf.reduce_mean(tf.reduce_sum(tf.losses.binary_crossentropy(true, pred), axis=-1))
# caculate accuracy
compute_accuracy = lambda true, pred: tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))
# Gradient Descent
optimizer = tf.optimizers.Adam(learning_rate)

for epoch in range(training_epochs):
    for i, (x_, y_) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            pred = model(x_)
            loss = compute_loss(y_, pred)
        acc = compute_accuracy(y_, pred)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        print("=> loss %.2f acc %.2f" %(loss.numpy(), acc.numpy()))