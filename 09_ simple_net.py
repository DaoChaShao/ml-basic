#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/8 22:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   09_ simple_net.py
# @Desc     :   

from numpy import array

from utils.D import numerical_gradient
from utils.network import SimpleNN
from utils.helper import generate_2d_metric


def f(model, W):
    model.params = W
    return model.forward(W)


def main() -> None:
    """ Main Function """
    # Initialise a piece of data
    features = generate_2d_metric(1, 2)
    X = features
    print(type(X), X.shape, X)
    # Initialise the label of the piece of data
    # - Assuming binary classification with multiple classes
    y = array([0, 0, 1])
    y_ture = y.argmax(axis=0)
    print(type(y), y.shape, y)
    print(type(y_ture), y_ture.shape, y_ture)

    # Initialise a simple neural network
    model = SimpleNN()
    y_prob = model.forward(X)
    y_pred = y_prob.argmax(axis=1)[0]
    print(y_prob, y_pred)
    print(type(y_pred), y_pred.shape)

    # Check if the prediction is correct
    if y_pred == y_ture:
        print("Correctly classified")
    else:
        print("Misclassified")
    CEE = model.loss(y, y_prob)
    print(CEE)

    # Compute the numerical gradient of the loss with respect to the weights and biases
    f = lambda _: model.loss(y, model.forward(X))
    grads = numerical_gradient(f, model.W)
    print(grads)


if __name__ == "__main__":
    main()
