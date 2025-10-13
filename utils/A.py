#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 12:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   A.py
# @Desc     :   

from numpy import (ndarray, array,
                   exp,
                   tanh as np_tanh,
                   maximum,
                   sum as np_sum,
                   max as np_max,
                   log as np_log,
                   arange, )

from utils.J import cross_entropy_error


def step(x: float) -> int:
    """ Binary Step Activation Function with scalar input
    :param x: value
    :return: 1 if x >= 0 else 0
    """
    return 1 if x >= 0 else 0


def binary_step(x: ndarray) -> ndarray:
    """ Binary Step Activation Function with array or vector input
    - Normally, it is used for classification tasks and in output layer
    :param x: numpy array
    :return: get 1 or 0
    """
    # Judge each element in the array:
    # - if >= 0 return True else return False
    # - then convert boolean array to int array, True -> 1, False -> 0
    return array(x >= 0, dtype=int)


def sigmoid(x: ndarray) -> ndarray:
    """ Sigmoid Activation Function with array or vector input
    - It is a smooth, S-shaped curve that maps any real-valued number to the (0, 1) interval.
    - It is commonly used for binary classification tasks and in output layer.
    - Range: -6 < h(x) < 6
    :param x: value
    :return: sigmoid value
    """
    return 1 / (1 + exp(-x))


def tanh(x: ndarray) -> ndarray:
    """ Hyperbolic Tangent (tanh) Activation Function with array or vector input
    - It is a smooth, S-shaped curve that maps any real-valued number to the (-1, 1) interval.
    - It is commonly used for classification tasks and in hidden layers of neural networks.
    - Range: -3 < h(x) < 3
    :param x: value
    :return: tanh value
    """
    return np_tanh(x)


def relu(x: ndarray) -> ndarray:
    """ Rectified Linear Unit (ReLU) Activation Function with array or vector input
    - It is defined as f(x) = max(0, x), meaning it outputs the input directly if it is positive; otherwise, it outputs zero.
    - It is widely used in hidden layers of deep neural networks due to its simplicity and effectiveness.
    - Range: 0 < h(x) < ∞
    :param x: value
    :return: ReLU value
    """
    return maximum(0, x)


def leaky_relu(x: ndarray, alpha: float = 0.01) -> ndarray:
    """ Leaky Rectified Linear Unit (Leaky ReLU) Activation Function with array or vector input
    - It is a variant of the ReLU function that allows a small, non-zero gradient when the input is negative.
    - It is defined as f(x) = x if x > 0 else alpha * x, where alpha is a small constant (e.g., 0.01).
    - It helps to mitigate the "dying ReLU" problem, where neurons can become inactive and only output zero.
    - Range: -∞ < h(x) < ∞
    :param x: value
    :param alpha: slope for negative input
    :return: Leaky ReLU value
    """
    return maximum(alpha * x, x)


def softmax(x: ndarray) -> ndarray:
    """ Softmax Activation Function with array or vector input
    - It is commonly used in the output layer of neural networks for multi-class classification tasks.
    - It converts raw scores (logits) into probabilities by exponentiating each score and normalizing them by the sum of all exponentiated scores.
    - The output values are in the range (0, 1) and sum to 1, making them interpretable as probabilities.
    :param x: value
    :return: softmax value
    """
    # Method II: Subtract the max for numerical stability to prevent overflow
    e_x = exp(x - np_max(x, axis=1, keepdims=True))
    return e_x / np_sum(e_x, axis=-1, keepdims=True)
    # Method I: Direct computation (may cause overflow for large values)
    return exp(x) / np_sum(exp(x))


def softMax(x: ndarray, axis: int = -1) -> ndarray:
    """ Softmax Activation Function for multidimensional arrays
    - It converts raw scores (logits) into probabilities by exponentiating each score
      and normalizing them by the sum of all exponentiated scores along the specified axis.
    - The output values are in the range (0, 1) and sum to 1 along the specified axis.

    Common usage:
    - For 1D arrays: axis=-1 (default)
    - For 2D arrays (batch_size, num_classes): axis=1 (apply to each sample)
    - For 3D+ arrays: specify the appropriate axis

    :param x: input array of any dimension
    :param axis: axis along which to apply softmax (default: -1, the last axis)
    :return: softmax probabilities with same shape as input
    """
    # Subtract the max for numerical stability to prevent overflow
    shifted_x = x - np_max(x, axis=axis, keepdims=True)
    exp_x = exp(shifted_x)
    return exp_x / np_sum(exp_x, axis=axis, keepdims=True)


def identity(x: ndarray) -> ndarray:
    """ Identity Activation Function with array or vector input
    - It is a linear activation function that outputs the input directly without any transformation.
    - It is commonly used in the output layer for regression tasks where the output can take any real value.
    - Range: -∞ < h(x) < ∞
    :param x: value
    :return: identity value
    """
    return x


class ReLU(object):
    """ ReLU Activation Function Class with forward and backward methods for use in neural networks """

    def __init__(self):
        self._mask: None | ndarray = None

    def forward(self, X: ndarray) -> ndarray:
        """ Forward pass for ReLU activation function
        :param X: input data
        :return: activated output
        """
        self._mask = (X <= 0).astype(bool)
        out = X.copy()
        out[self._mask] = 0
        return out

    def backward(self, derivative_of_output: ndarray) -> ndarray:
        """ Backward pass for ReLU activation function
        :param derivative_of_output: gradient from the next layer, which can be named dL/dout also.
        :return: gradient with respect to the input of ReLU
        """
        dx = derivative_of_output.copy()
        dx[self._mask] = 0
        return dx


class Sigmoid(object):
    """ Sigmoid Activation Function Class with forward and backward methods for use in neural networks """

    def __init__(self):
        self._out = None

    def forward(self, X: ndarray) -> ndarray:
        """ Forward pass for Sigmoid activation function
        :param X: input data
        :return: activated output
        """
        self._out = 1 / (1 + exp(-X))
        return self._out

    def backward(self, derivative_of_output: ndarray) -> ndarray:
        """ Backward pass for Sigmoid activation function
        :param derivative_of_output: gradient from the next layer, which can be named dL/dout also.
        :return: gradient with respect to the input of Sigmoid
        """
        dx = derivative_of_output * (1.0 - self._out) * self._out
        return dx


class Affine(object):
    """ Affine (Fully Connected) Layer Class with forward and backward methods for use in neural networks """

    def __init__(self, W: ndarray, b: ndarray):
        self._W = W
        self._b = b

        self._X = None
        self._X_shape = None

        self._dW = None
        self._db = None

    def forward(self, X: ndarray) -> ndarray:
        """ Forward pass for Affine layer
        :param X: input data
        :return: output after affine transformation
        """
        self._X_shape = X.shape
        # Flatten input from 3d tensor to 2d array if necessary
        self._X = X.reshape(self._X_shape[0], -1)
        out = X.dot(self._W) + self._b
        return out

    def backward(self, derivative_of_output: ndarray) -> ndarray:
        """ Backward pass for Affine layer
        :param derivative_of_output: gradient from the next layer, which can be named dL/dout also.
        :return: gradients with respect to input, weights, and biases
        """
        dX = derivative_of_output.dot(self._W.T)
        # Reshape dx to the original input shape
        dX = dX.reshape(*self._X_shape)

        self._dW = self._X.T.dot(derivative_of_output)
        self._db = np_sum(derivative_of_output, axis=0)

        return dX


class SoftmaxWithLoss(object):
    """ Softmax Activation Function combined with Cross-Entropy Loss Class
    - This class combines the softmax activation function and the cross-entropy loss function into a single layer.
    - It is commonly used in the output layer of neural networks for multi-class classification tasks.
    - The forward method computes the softmax probabilities and the cross-entropy loss.
    - The backward method computes the gradient of the loss with respect to the input scores.
    """

    def __init__(self):
        self._loss = None
        # Predicted probabilities after softmax
        self._pred = None
        # True labels (one-hot encoded)
        self._target = None

    def forward(self, X: ndarray, y_true: ndarray) -> float:
        """ Forward pass for Softmax with Cross-Entropy Loss
        :param X: input scores (logits)
        :param y_true: true labels (one-hot encoded)
        :return: cross-entropy loss
        """
        self._target = y_true
        self._pred = softmax(X)
        self._loss = cross_entropy_error(self._pred, self._target)
        return self._loss

    def backward(self, derivative_of_output: float = 1.0) -> ndarray:
        """ Backward pass for Softmax with Cross-Entropy Loss
        :param derivative_of_output: gradient from the next layer, which can be named dL/dout also. Default is 1.
        :return: gradient with respect to the input scores
        """
        batch_size = self._target.shape[0]
        if self._target.size == self._pred.size:
            # one-hot label
            dx = (self._pred - self._target) / batch_size
        else:
            # class indices
            dx = self._pred.copy()
            dx[arange(batch_size), self._target] -= 1
            dx = dx / batch_size
        return dx
