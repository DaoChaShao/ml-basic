#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 12:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_tanh.py
# @Desc     :   

from utils.A import tanh
from utils.helper import generate_1d_array


def main() -> None:
    """ Main Function """
    # Generate a random integer array with values between -4 and 6, size 10
    arr = generate_1d_array(-4, 6, 10)
    print(f"Random Integer Array:\n{arr}")

    # Apply the tanh activation function to the array
    values = tanh(arr)
    print(f"Tanh Values:\n{values}")


if __name__ == "__main__":
    main()
