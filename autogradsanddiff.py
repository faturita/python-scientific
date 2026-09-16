"""
====================================
Autograd: Automatic Differentiation
====================================

Based on https://sscardapane.github.io/learn-autodiff/

Autograd and autodiff are key componentes in tensorflow and other libraries because it allows automatic differentiation for gradient
descent.

"""


print(__doc__)

import autograd.numpy as np
from autograd import grad

# Recurrent expansion of the taylor series of sine
def taylor_sine(x):
    ans = currterm = x
    i = 0
    while np.abs(currterm) > 0.001:
        currterm = -currterm * x**2 / ((2 * i +3) * (2 * i +2))
        ans = ans + currterm
        i += 1
    return ans


grad_sine = grad(taylor_sine)

# So the grad of sine is cosine
print(grad_sine(0.5), np.cos(0.5))


# Logistic regression with autograd 
# (This is non linear perceptron with a sigmoid activation function a binary cross entropy loss)

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to the logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)

    # This is numerically more stable but equivalent to:  np.log( (preds ** targets)  ( (1 - preds) ** (1 - targets) )
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
print("Initial loss:", training_loss(weights))
for i in range(10000):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))


for i in range(4):
    print( "logistic_predictions(weights, inputs[{}]) = {}".format(i, logistic_predictions(weights, inputs[i])) )