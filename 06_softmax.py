#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 13:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   06_softmax.py
# @Desc     :   

from numpy import array
from utils.A import softmax, softMax
from utils.helper import generate_1d_array, generate_3d_metric


def main() -> None:
    """ Main Function """
    # Generate a random integer array
    arr = generate_1d_array(-10, 10, 10)
    print(f"Random Integer Array:\n{arr}")

    # Apply the Sigmoid activation function
    values = softmax(arr)
    print(f"Softmax Activation Function Result:\n{values}")
    print(f"Sum of Softmax Values: {values.sum()}")

    arr_1d = generate_1d_array(-10, 10, 4)
    print(f"Random Integer 1D Array:\n{arr_1d}")
    values_1d = softMax(arr_1d)
    print(f"SoftMax Activation Function Result:\n{values_1d}")
    print(f"Sum of SoftMax Values: {values_1d.sum()}")

    arr_2d = generate_3d_metric(1, 2, 3)
    print(f"Random Integer 2D Array:\n{arr_2d}")
    values_2d = softMax(arr_2d, axis=1)
    print(f"SoftMax Activation Function Result:\n{values_2d}")
    print(f"Sum of SoftMax Values (axis=1): {values_2d.sum(axis=1)}")

    arr_3d = generate_3d_metric(2, 3, 4)
    print(f"Random Float 3D Array:\n{arr_3d}")
    values_3d = softMax(arr_3d, axis=-1)
    print(f"SoftMax Activation Function Result:\n{values_3d}")
    print(f"Sum of SoftMax Values (axis=2): {values_3d.sum(axis=-1)}")


if __name__ == "__main__":
    main()
