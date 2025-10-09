#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 16:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   J.py
# @Desc     :

from numpy import sum as np_sum, clip, log, mean as np_mean, ndarray, zeros_like


def mean_squared_error(pred, target) -> float:
    """ Calculate Mean Squared Error (MSE) between true and predicted values """
    return 0.5 * np_sum((pred - target) ** 2)


def cross_entropy_error(pred, target) -> float:
    """ Calculate Cross-Entropy Error between true and predicted values (with clipping for stability)
    :param pred: predicted probabilities for each class (values between 0 and 1,
    :param target: true class labels (as class indices or one-hot encoded)
    :return: cross-entropy error
    """
    if pred.ndim == 1:
        # Reshape 1D array to 2D array with one row
        pred = pred.reshape(1, pred.size)
        target = target.reshape(1, target.size)
    # If target is one-hot encoded, convert to class indices
    if target.size == pred.size:
        target = target.argmax(axis=1)
    # number of samples in the batch
    n = pred.shape[0]
    return -np_sum(log(pred[range(n), target] + 1e-7)) / n
