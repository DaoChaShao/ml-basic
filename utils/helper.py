#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 12:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :   

from numpy import random as np_random
from numpy import ndarray, array


def generate_1d_array(low: int, high: int, size: int) -> ndarray:
    """ Generate a random integer array
    :param low: lower bound (inclusive)
    :param high: upper bound (exclusive)
    :param size: size of the array
    :return: numpy array of random integers
    """
    return np_random.randint(low, high, size)


def generate_2d_metric(rows: int, cols: int) -> ndarray:
    """ Generate a random float tensor (2D array) with standard normal distribution values (mean=0, std=1) """
    return np_random.randn(rows, cols)


def generate_3d_metric(dims: int, rows: int, cols: int) -> ndarray:
    """ Generate a random float tensor (3D array) with standard normal distribution values (mean=0, std=1) """
    return np_random.randn(dims, rows, cols)
