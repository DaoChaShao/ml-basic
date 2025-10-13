#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/13 21:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   optimiser.py
# @Desc     :

from numpy import zeros_like, sqrt


class SGD(object):
    """ Stochastic Gradient Descent (SGD) Optimizer
    - Performs parameter updates using the formula: param -= learning_rate * gradient
    - Suitable for large datasets and online learning scenarios
    """

    def __init__(self, alpha: float = 0.01):
        self._learning_rate = alpha

    def update(self, params: dict, grads: dict):
        """ Update parameters using SGD
        :param params: dictionary of model parameters (weights and biases)
        :param grads: dictionary of gradients for each parameter
        """
        for key in params.keys():
            params[key] -= self._learning_rate * grads[key]


class Momentum(object):
    """ Momentum Optimizer
    - Accelerates SGD by adding a momentum term, which is historical gradients, to the parameter updates
    - Helps to smooth out updates and can lead to faster convergence
    """

    def __init__(self, alpha: float = 0.01, beta: float = 0.9):
        self._learning_rate = alpha
        self._momentum = beta
        self._velocity = None

    def update(self, params: dict, grads: dict):
        """ Update parameters using Momentum
        :param params: dictionary of model parameters (weights and biases)
        :param grads: dictionary of gradients for each parameter
        """
        if self._velocity is None:
            self._velocity = {key: 0 for key in params.keys()}

        for key in params.keys():
            self._velocity[key] = self._momentum * self._velocity[key] - self._learning_rate * grads[key]
            params[key] += self._velocity[key]


class AdaGrad(object):
    """ AdaGrad Optimizer
    - Adapts the learning rate for each parameter based on the historical sum of squared gradients
    - Helps to perform larger updates for infrequent parameters and smaller updates for frequent parameters
    """

    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-7):
        self._learning_rate = alpha
        self._epsilon = epsilon
        # historical squared gradients can be named as h
        self._h = None

    def update(self, params: dict, grads: dict):
        """ Update parameters using AdaGrad
        :param params: dictionary of model parameters (weights and biases)
        :param grads: dictionary of gradients for each parameter
        """
        self._h = {}
        for key, val in params.items():
            self._h[key] = zeros_like(val)

        for key in params.keys():
            self._h[key] += grads[key] * grads[key]
            params[key] -= self._learning_rate * grads[key] / (sqrt(self._h[key] + self._epsilon))


class RMSProp(object):
    """ RMSProp Optimizer
    - Adapts the learning rate for each parameter based on a moving average of squared gradients
    - Helps to maintain a more stable learning rate and can lead to better convergence
    - EMA (Exponential Moving Average) is used to compute the moving average of squared gradients
    """

    def __init__(self, alpha: float = 0.001, beta: float = 0.9, epsilon: float = 1e-7):
        self._learning_rate = alpha
        self._decay_rate = beta
        self._epsilon = epsilon
        self._h = None

    def update(self, params: dict, grads: dict):
        """ Update parameters using RMSProp
        :param params: dictionary of model parameters (weights and biases)
        :param grads: dictionary of gradients for each parameter
        """
        if self._h is None:
            self._h = {key: zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self._h[key] *= self._decay_rate
            self._h[key] += (1 - self._decay_rate) * grads[key] * grads[key]
            params[key] -= self._learning_rate * grads[key] / (sqrt(self._h[key]) + self._epsilon)
