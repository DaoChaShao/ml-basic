#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 14:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   network.py
# @Desc     :

from collections import OrderedDict
from numpy import random as np_random, zeros, ndarray, sum as np_sum, argmax
from typing import Union

from utils.A import sigmoid, softmax, identity, Affine, ReLU, SoftmaxWithLoss
from utils.D import numerical_gradient
from utils.J import cross_entropy_error


def network_init(
        features: int,
        hidden_inner_units: int,
        hidden_outer_units: int,
        output_units: int,
        init_scale: float = 0.01
) -> dict:
    """ Initialize a 3-layer neural network with random weights and zero biases
    :param features: Number of input features
    :param hidden_inner_units: Number of neurons in the first hidden layer
    :param hidden_outer_units: Number of neurons in the second hidden layer
    :param output_units: Number of output units
    :param init_scale: Scale for random weight initialization
    :return: Dictionary containing weights and biases for each layer
    """
    parameters: dict = {
        # Initialize weights and biases for the first layer
        "W1": np_random.randn(features, hidden_inner_units) * init_scale,
        "b1": zeros((1, hidden_inner_units)),
        # Initialize weights and biases for the second layer
        "W2": np_random.randn(hidden_inner_units, hidden_outer_units) * init_scale,
        "b2": zeros((1, hidden_outer_units)),
        # Initialize weights and biases for the output layer
        "W3": np_random.randn(hidden_outer_units, output_units) * init_scale,
        "b3": zeros((1, output_units))
    }

    return parameters


def forward(X, parameters: dict) -> tuple[ndarray, dict]:
    """ Perform forward propagation through the network
    :param X: Input data
    :param parameters: Dictionary containing weights and biases for each layer
    :return: Dictionary containing activations for each layer
    """
    # Retrieve weights and biases from parameters
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Compute activations for the inner hidden layer
    Z1 = X.dot(W1) + b1
    # Activation function can be applied
    A1 = sigmoid(Z1)
    # Compute activations for the outer hidden layer
    Z2 = A1.dot(W2) + b2
    # Activation function can be applied
    A2 = sigmoid(Z2)
    # Compute activations for the output layer
    Z3 = A2.dot(W3) + b3
    # Activation function can be applied
    # - Method I: for regression tasks, use identity function
    A3 = identity(Z3)
    # - Method II: for regression tasks, no activation function
    # A3 = Z3

    # Store activations for each batch in a dictionary; discard it when the batch processing is complete
    cache_per_batch: dict = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3
    }

    return A3, cache_per_batch


class SimpleNN(object):
    """ Minimal single-layer neural network for 2-feature, 3-class classification """

    def __init__(self, scale: float = 0.01):
        # Initialize weights, small random values and units for 2 input features and 3 output classes
        self.W = np_random.randn(2, 3) * scale
        self._y_pred: Union[ndarray | None] = None
        self._cee = 0.0

    def forward(self, X: ndarray) -> ndarray:
        Z = X.dot(self.W)
        A = softmax(Z)
        # Activation result is the final prediction
        self._y_pred = A
        return self._y_pred

    @staticmethod
    def loss(y_true: ndarray, y_pred: ndarray) -> float:
        """ Compute Mean Squared Error loss
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: MSE loss value
        """
        return cross_entropy_error(y_true, y_pred)


class Simple2LayersNN(object):
    """ Minimal two-layer neural network for 2-feature, 3-class classification """

    def __init__(self, input_size: int, hidden_units: int, output_size: int, scale: float = 0.01):
        self._parameters: dict = {
            # Initialize weights and biases for the first layer
            "W1": np_random.randn(input_size, hidden_units) * scale,
            "b1": zeros((1, hidden_units)),
            # Initialize weights and biases for the output layer
            "W2": np_random.randn(hidden_units, output_size) * scale,
            "b2": zeros((1, output_size))
        }

    def forward(self, X: ndarray) -> ndarray:
        W1, b1 = self._parameters["W1"], self._parameters["b1"]
        W2, b2 = self._parameters["W2"], self._parameters["b2"]

        Z1 = X.dot(W1) + b1
        A1 = sigmoid(Z1)
        Z2 = A1.dot(W2) + b2

        return softmax(Z2)

    def loss(self, X: ndarray, t: ndarray) -> float:
        """ Compute Mean Squared Error loss
        :param X : Input data
        :param t: True labels
        :return: CEE loss value
        """
        y_pred = self.forward(X)
        return cross_entropy_error(y_pred, t)

    def grad_descent(self, X, t):
        fn = lambda p: self.loss(X, t)
        grads = {
            "W1": numerical_gradient(fn, self._parameters["W1"]),
            "b1": numerical_gradient(fn, self._parameters["b1"]),
            "W2": numerical_gradient(fn, self._parameters["W2"]),
            "b2": numerical_gradient(fn, self._parameters["b2"]),
        }
        return grads

    def getter(self):
        return self._parameters

    @staticmethod
    def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
        """ Compute accuracy of predictions
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Accuracy value
        """
        y_true_labels = y_true.argmax(axis=1)
        y_pred_labels = y_pred.argmax(axis=1)
        return np_sum(y_true_labels == y_pred_labels) / len(y_true_labels)


class Full2LayersNN(object):
    """ Minimal two-layer neural network for 2-feature, 3-class classification """

    def __init__(self, input_size: int, hidden_units: int, output_size: int, scale: float = 0.01):
        # Initialize weights, small random values and units for 2 input features and 3 output classes
        self._parameters: dict = {
            # Initialize weights and biases for the first layer
            "W1": np_random.randn(input_size, hidden_units) * scale,
            "b1": zeros((1, hidden_units)),
            # Initialize weights and biases for the output layer
            "W2": np_random.randn(hidden_units, output_size) * scale,
            "b2": zeros((1, output_size))
        }

        # Initialize layers in an ordered dictionary to maintain the sequence of operations
        self._layers: OrderedDict = OrderedDict()
        self._layers["Affine1"] = Affine(self._parameters["W1"], self._parameters["b1"])
        self._layers["ReLU1"] = ReLU()
        self._layers["Affine2"] = Affine(self._parameters["W2"], self._parameters["b2"])
        self._lastLayer = SoftmaxWithLoss()

    def forward(self, X: ndarray) -> ndarray:
        for layer in self._layers.values():
            X = layer.forward(X)
        return X

    def loss(self, X: ndarray, t: ndarray) -> float:
        """ Compute Mean Squared Error loss
        :param X : Input data
        :param t: True labels
        :return: CEE loss value
        """
        y_pred = self.forward(X)
        return self._lastLayer.forward(y_pred, t)

    def grad_descent(self, X, t):
        fn = lambda p: self.loss(X, t)
        grads = {
            "W1": numerical_gradient(fn, self._parameters["W1"]),
            "b1": numerical_gradient(fn, self._parameters["b1"]),
            "W2": numerical_gradient(fn, self._parameters["W2"]),
            "b2": numerical_gradient(fn, self._parameters["b2"]),
        }
        return grads

    def getter(self):
        return self._parameters

    def accuracy(self, X: ndarray, target: ndarray) -> float:
        """ Compute accuracy of predictions
        :param X: Input data
        :param target: True labels
        :return: Accuracy value
        """
        y_prob = self.forward(X)
        y_pred = argmax(y_prob, axis=1)

        if target.ndim != 1:
            target = argmax(target, axis=1)

        return np_sum(y_pred == target) / X.shape[0]

    def backward(self, X: ndarray, target: ndarray, dout: float = 1.0):
        """ Backward pass through the network to compute gradients
        :param X: Input data
        :param target: True labels
        :param dout: Gradient from the next layer, default is 1.0
        :return: Gradients with respect to weights and biases
        """
        # Compute loss to set up for backward pass
        self.loss(X, target)

        # Start backward pass from the last layer
        dy = self._lastLayer.backward(dout)
        # Backpropagate through each layer in reverse order
        for layer in reversed(list(self._layers.values())):
            dy = layer.backward(dy)

        # Collect gradients from each layer
        grads = {
            "W1": self._layers["Affine1"]._dW, "b1": self._layers["Affine1"]._db,
            "W2": self._layers["Affine2"]._dW, "b2": self._layers["Affine2"]._db
        }
        return grads
