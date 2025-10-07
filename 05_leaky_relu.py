#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 13:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   05_leaky_relu.py
# @Desc     :   

from random import uniform
from utils.A import leakyReLU
from utils.helper import generate_1d_array


def main() -> None:
    """ Main Function """
    # Generate a random integer array
    arr = generate_1d_array(-10, 10, 10)
    print(f"Input Array:\n{arr}")

    # Apply Leaky ReLU activation function
    alpha = uniform(0, 0.03)
    print(f"Leaky ReLU alpha: {alpha:.4f}")
    values = leakyReLU(arr, alpha=0.01)
    print(f"Leaky ReLU Activated Array:\n{values}")


if __name__ == "__main__":
    main()
