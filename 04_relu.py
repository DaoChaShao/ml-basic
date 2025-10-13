#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 13:06
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   04_relu.py
# @Desc     :   

from utils.A import relu
from utils.helper import generate_1d_array


def main() -> None:
    """ Main Function """
    # Generate a random integer array
    arr = generate_1d_array(-10, 10, 10)
    print(f"Input Array:\n{arr}")

    # Apply ReLU activation function
    values = relu(arr)
    print(f"ReLU Activated Array:\n{values}")


if __name__ == "__main__":
    main()
