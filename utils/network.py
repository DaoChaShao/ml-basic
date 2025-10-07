#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 14:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   network.py
# @Desc     :   

from numpy import random as np_random, zeros, ndarray

from utils.A import sigmoid, identity


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


class NetworkFor3Layers:
    """ A simple 3-layer neural network class """

    def __init__(
            self,
            features: int,
            hidden_inner_units: int,
            hidden_outer_units: int,
            output_units: int,
            init_scale: float = 0.01
    ) -> None:
        """ Initialize the neural network with random weights and zero biases
        :param features: Number of input features
        :param hidden_inner_units: Number of neurons in the first hidden layer
        :param hidden_outer_units: Number of neurons in the second hidden layer
        :param output_units: Number of output units
        :param init_scale: Scale for random weight initialization
        """
        self._parameters: dict = {
            # Initialize weights and biases for the first layer
            "W1": np_random.randn(features, hidden_inner_units) * init_scale,
            "b1": zeros(1, hidden_inner_units),
            # Initialize weights and biases for the second layer
            "W2": np_random.randn(hidden_inner_units, hidden_outer_units) * init_scale,
            "b2": zeros(1, hidden_outer_units),
            # Initialize weights and biases for the output layer
            "W3": np_random.randn(hidden_outer_units, output_units) * init_scale,
            "b3": zeros(1, output_units)
        }

    def forward(self, X) -> tuple[ndarray, dict]:
        """ Perform forward propagation through the network
        :param X: Input data
        :return: Dictionary containing activations for each layer
        """
        # Retrieve weights and biases from parameters
        W1, b1 = self._parameters["W1"], self._parameters["b1"]
        W2, b2 = self._parameters["W2"], self._parameters["b2"]
        W3, b3 = self._parameters["W3"], self._parameters["b3"]

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
