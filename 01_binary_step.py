#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 11:38
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   01_binary_step.py
# @Desc     :   

from utils.A import binary_step
from utils.helper import generate_1d_array


def main() -> None:
    """ Main Function """
    # Generate a random integer array between -5 and 5 with size 10
    arr = generate_1d_array(-5, 5, 10)
    print(f"Generated Array:\n{arr}")

    # Apply Binary Step Activation Function
    values = binary_step(arr)
    print(f"Classification Result:\n{values}")


if __name__ == "__main__":
    main()
