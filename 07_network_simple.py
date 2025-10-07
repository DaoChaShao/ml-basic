#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/7 14:24
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   07_network_simple.py
# @Desc     :   

from pprint import pprint
from utils.helper import generate_2d_metric
from utils.network import network_init, forward


def main() -> None:
    """ Main Function """
    # Generate a random input array with shape (3, 4, 5) for batch processing
    data = generate_2d_metric(3, 4)
    print(f"Input Data Shape: {data.shape}")

    # Initialize a simple neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    parameters = network_init(data.shape[1], 128, 64, 2)
    pprint(parameters)

    # Perform a forward pass through the network
    activation, cache = forward(data, parameters)
    print(f"Output Data after Forward Pass: {activation}")
    pprint(cache)


if __name__ == "__main__":
    main()
