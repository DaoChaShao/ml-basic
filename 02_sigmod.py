#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 12:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   02_sigmod.py
# @Desc     :   

from utils.A import sigmoid
from utils.helper import generate_1d_array


def main() -> None:
    """ Main Function """
    # Generate a random integer array with values between -3 and 4, size 10
    arr = generate_1d_array(-5, 6, 10)
    print(f"Random Integer Array:\n{arr}")

    # Apply the sigmoid activation function to the array
    values = sigmoid(arr)
    print(f"Sigmoid Values:\n{values}")


if __name__ == "__main__":
    main()
