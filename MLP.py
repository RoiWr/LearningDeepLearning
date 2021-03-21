""" Multi-level perceptron (MLP):
2 fully-connected layers neural network from scratch
-- ITC NN-rolling #1 assignment --
by Roi Weinberger. Date: January 2020
_________________________________________
This script contains the class Network that creates instances of MLPs utilizing the following additional classes:
1) Layer class: composed of 2 subclasses
    a. the Linear class
    b. activation function: one of the following classes:
        i. ReLU
        ii. Identity
        iii. Sigmoid
        iv. Softmax

2) Loss functions: one of the following classes:
    a. MSE
    b. binary cross-entropy

The Network is optimized by stochastic gradient descent (SGD),
running each epoch on a random mini-batch of the train set.
"""

import numpy as np

# activation functions
class ReLU:
    """ ReLU activation function for perceptron layer """
    def forward(self, x):
        self.x = x
        x[x < 0] = 0
        return x

    def backward(self, dz):
        dz[self.x < 0] = 0
        return dz

class Sigmoid:
    """ Sigmoid activation function for perceptron layer """
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dz):
        dx = np.exp(-self.x) / ((1 + np.exp(-self.x)) ** 2)
        return np.multiply(dx, dz)

class Softmax:
    """ Softmax activation function for perceptron layer """
    def forward(self, x):
        self.x = x
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def backward(self, dz):
        dx = self.forward(self.x) - self.forward(self.x) ** 2
        return dx * dz

class Identity:
    """ Identity activation function for perceptron layer """
    def forward(self, x):
        self.x = x
        return x

    def backward(self, dz):
        return dz

# Linear class
class Linear:
    """ Class for the linear functions of a perceptron layer. """
    def forward(self, X, W, b):
        """ multiplication of X (input) by W (weight matrix) followed by addition of b (bias vector) """
        self.X = X
        self.W = W
        self.b = b
        return np.add(np.matmul(X, W), b)

    def backward(self, dz):
        """ calculate the gradients using the chain rule for W, X, b, given back-propagated gradient dz"""
        dW = np.matmul(self.X.T, dz)
        db = np.sum(dz, axis=0).reshape(-1, 1)
        dX = np.matmul(dz, self.W.T)
        return {'dW': dW, 'db': db, 'dX': dX}

# Layer class
class Layer:
    """ The perceptron layer. Composed of a linear part (Linear class) and a given activation function """
    def __init__(self, func, n_in, n_out):
        """ The class is initialized by the given activation function (func) and number of features ("neurons")
        of the input (n_in) and of the output (n_out) """
        self.Linear = Linear()
        self.Activation = func()
        self.grad = None
        limit = 1
        self.W = np.random.uniform(-limit, limit, (n_in, n_out))  # W has dimensions of [n_in, n_out]
        self.b = np.random.uniform(-limit, limit, (1, n_out))  # b has dimensions of [1, n_out]

    def forward(self, X):
        lin_output = self.Linear.forward(X, self.W, self.b)
        layer_output = self.Activation.forward(lin_output)
        return layer_output

    def backward(self, dz):
        activation_grad = self.Activation.backward(dz)
        self.grad = self.Linear.backward(activation_grad)
        pass

# Loss functions
class binary_cross_entropy:
    """ Class for binary cross-entropy loss function calculation and gradient """
    def calculate(self, true, pred):
        pred[pred <= 0.] = 1e-7
        return (-1 / len(true)) * sum(true * np.log(pred) + (1 - true) * np.log(1 - pred))

    def gradient(self, true, pred):
        grad = np.zeros_like(pred)
        for i in range(len(pred)):
            y_i = true[i].astype(float)
            if pred[i] == 0.:
                p = 1e-7
            else:
                p = pred[i]
            grad[i] = (-1 / len(true)) * (y_i / p) + (-(1 - y_i) / (1 - p))
        return grad

class MSE:
    """ Class for Mean Square Error (MSE) calculation and gradient """
    def calculate(self, true, pred):
        """ Calculates MSE """
        if len(true) != len(pred):
            raise ValueError('true and pred are of different lengths')
        return (1 / len(true)) * sum((true - pred) ** 2)

    def gradient(self, true, pred):
        """ Returns the gradient of the MSE. The gradient is calculated per pred[i]"""
        if len(true) != len(pred):
            raise ValueError('true and pred are of different lengths')
        # the gradient is the derivative per each y_pred[i]
        return (2 / len(true)) * -(true - pred)

class Network:
    """ This class creates instances of MLPs utilizing the following additional classes:
1) Layer class: composed of 2 subclasses
    a. the Linear class
    b. activation function: one of the following classes:
2) Loss functions.
The Network is optimized by stochastic gradient descent (SGD),
running each epoch on a random mini-batch of the train set. """
    def __init__(self, n_features, n_classes, layer1_function, layer1_n_out, layer2_function, loss_func=MSE,
                 a=1e-3, mini_batch_size=2, max_runs=10):
        """ n_features = the number of features of the input (size of X[i])
        n_classes = the number of classes in the input (size of y[i])
        layer1/2_function = the activation function (not as string) for that layer
        layer_1_n_out = the size of the hidden layer
        loss_func = the loss function
        a = the learning rate
        mini_batch_size = (m) the size of the mini-batch (sub-sample of X) in each epoch
        max_runs = (t) the number of epochs to run the network
        """
        self.l1 = Layer(layer1_function, n_in=n_features, n_out=layer1_n_out)
        self.l2 = Layer(layer2_function, n_in=layer1_n_out, n_out=n_classes)
        self.loss = loss_func()
        self.a = a  # learning rate
        self.t = max_runs  # max number of epochs to run
        self.m = mini_batch_size
        self.error = None

    def optimizer(self, W, b, dW, db):
        """ Gradient descent optimizer """
        new_W = W - self.a * dW
        new_b = b - self.a * db.T
        return [new_W, new_b]

    def mini_batch(self, X, y):
        """ Randomly samples m samples from X (the input) to run each epoch. For stochastic gradient descent"""
        idx = np.random.choice(range(len(y)), size=self.m, replace=True)
        return X[idx, :], y[idx]

    def forward(self, x):
        """ Forward propagation of the network for each layer """
        l1_output = self.l1.forward(x)
        l2_output = self.l2.forward(l1_output)
        return l2_output

    def backward(self, y_true, y_pred):
        """ Backward propagation of the netwoek, returning the gradient and updating weights """
        # estimate gradient for loss function
        dL = self.loss.gradient(y_true, y_pred)
        # calculate gradient for layer 2 and update values
        self.l2.backward(dL)
        [self.l2.W, self.l2.b] = self.optimizer(self.l2.W, self.l2.b, self.l2.grad['dW'], self.l2.grad['db'])
        # calculate gradient for layer 1 and update values
        self.l1.backward(self.l2.grad['dX'])
        #print(self.l1.b)
        [self.l1.W, self.l1.b] = self.optimizer(self.l1.W, self.l1.b, self.l1.grad['dW'], self.l1.grad['db'])

    def get_error(self, y_true, y_pred):
        """ Calculates the error given the loss function """
        return self.loss.calculate(y_true, y_pred)

    def fit(self, X, y):
        """ Fit function for this model given input X and output y """
        i = 0
        while i < self.t:
            xb, yb = self.mini_batch(X, y)  # mini-batch sample
            # xb = X
            # yb = y
            pred = self.forward(xb)
            self.backward(y_true=yb, y_pred=pred)
            i += 1

    def predict(self, sample):
        """ Run a sample through the model to get a prediction """
        return self.forward(sample)