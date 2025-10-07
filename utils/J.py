#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 16:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   J.py
# @Desc     :

from numpy import sum as np_sum, clip, log, mean as np_mean


def mean_squared_error(pred, target) -> float:
    """ Calculate Mean Squared Error (MSE) between true and predicted values """
    return 0.5 * np_sum((pred - target) ** 2)


def cross_entropy_error_binary(pred, target) -> float:
    """ Calculate Binary Cross-Entropy Error between true and predicted values (with clipping for stability)
    :param pred: predicted probabilities (values between 0 and 1)
    :param target: true binary labels (0 or 1)
    :return: binary cross-entropy error
    """
    # Avoid log(0) by clipping predictions to a small range
    eps = 1e-12
    pred = clip(pred, eps, 1 - eps)
    loss = - (target * log(pred) + (1 - target) * log(1 - pred))
    return float(np_mean(loss))


def cross_entropy_error_categorical(pred, target) -> float:
    """ Calculate Multi-Class Cross-Entropy Error between true and predicted values (with clipping for stability)
    :param pred: predicted probabilities for each class (values between 0 and 1, sum to 1)
    :param target: true class labels (one-hot encoded)
    :return: multi-class cross-entropy error
    """
    # Avoid log(0) by clipping predictions to a small range
    eps = 1e-12
    pred = clip(pred, eps, 1 - eps)
    loss = - np_sum(target * log(pred), axis=1)
    return float(np_mean(loss))
