#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/8 21:43
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   D.py
# @Desc     :   

from numpy import ndarray, zeros_like, array


def central_diff(f, x, h: float = 1e-5) -> float:
    """ Numerical Forward Differentiation (NOT reverse-mode differentiation)
    - Compute the numerical derivative of function f at point x using central difference method
    - Slope of the function at point x approximated using finite differences
    :param f: function to compute the gradient for
    :param x: point at which to compute the gradient
    :param h: small step size for finite difference
    :return: numerical gradient at point x
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def exact_gradient(x):
    """ Analytical Forward Differentiation (NOT reverse-mode differentiation)
    - Compute the exact derivative of the example function at point x using calculus
    - Slope of the function at point x exactly computed using calculus
    :param x: point at which to compute the derivative
    :return:  value at point x
    """
    return 0.02 * x + 0.1


def central_gradient(f, weights: ndarray, h: float = 1e-5) -> ndarray:
    """ Numerical Forward Differentiation for multidimensional arrays
    - Compute the numerical gradient of function f at point x using central difference method
    - Slope of the function at point x approximated using finite differences
    :param f: function to compute the gradient for
    :param weights: weights of the model (array) or array of points at which to compute the gradient
    :param h: small step size for finite difference
    :return: numerical gradient array at point x
    """
    # Initialize gradient array with the same shape as x. The initial values are zeros, which is not important.
    grad = zeros_like(weights)
    for i in range(weights.size):
        # copy x to avoid modifying the original array
        x_h1 = weights.copy()
        x_h2 = weights.copy()

        # central difference
        x_h1[i] += h
        x_h2[i] -= h
        grad[i] = (f(x_h1) - f(x_h2)) / (2 * h)

    return grad


def numerical_gradient(f, WEIGHTS: ndarray, h: float = 1e-5) -> ndarray:
    """ Numerical Forward Differentiation for multidimensional arrays
    - Compute the numerical gradient of function f at point x using central difference method
    - Slope of the function at point x approximated using finite differences
    :param f: function to compute the gradient for
    :param WEIGHTS: weights of the model (array) or array of points at which to compute the gradient
    :param h: small step size for finite difference
    :return: numerical gradient array at point x
    """
    if WEIGHTS.ndim == 1:
        return central_gradient(f, WEIGHTS, h)
    else:
        # Initialize gradient array with the same shape as x. The initial values are zeros, which is not important.
        grad = zeros_like(WEIGHTS)
        for i in range(WEIGHTS.shape[0]):
            grad[i] = central_gradient(f, WEIGHTS[i], h)
        return grad


def gradient_descent(f, init_x: ndarray, lr: float = 0.01, step_num: int = 100) -> tuple[ndarray, ndarray]:
    """ Gradient Descent Optimization
    - Perform gradient descent to minimize function f starting from initial point init_x
    - Update the point iteratively by moving in the direction of the negative gradient scaled by learning rate
    :param f: function to minimize
    :param init_x: initial point (array)
    :param lr: learning rate (step size for each update), which is called alpha in some contexts
    :param step_num: number of iterations to perform
    :return: final point after optimization and history of points visited
    """
    x = init_x
    f_values: list = []

    for i in range(step_num):
        f_values.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, array(f_values)
