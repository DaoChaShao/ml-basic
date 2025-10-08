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
                   max as np_max)


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


def ReLU(x: ndarray) -> ndarray:
    """ Rectified Linear Unit (ReLU) Activation Function with array or vector input
    - It is defined as f(x) = max(0, x), meaning it outputs the input directly if it is positive; otherwise, it outputs zero.
    - It is widely used in hidden layers of deep neural networks due to its simplicity and effectiveness.
    - Range: 0 < h(x) < ∞
    :param x: value
    :return: ReLU value
    """
    return maximum(0, x)


def leakyReLU(x: ndarray, alpha: float = 0.01) -> ndarray:
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
    e_x = exp(x - np_max(x))
    return e_x / np_sum(e_x, axis=-1)
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
