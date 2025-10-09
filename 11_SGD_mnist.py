#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/9 17:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_SGD_mnist.py
# @Desc     :   

from numpy import random as np_random, ceil
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.datasets import mnist
from time import perf_counter

from utils.network import TwoLayersNN


def mnist_loader():
    """ Load MNIST dataset """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(type(x_train), x_train.shape, x_train.dtype)
    print(type(y_train), y_train.shape, y_train.dtype)
    print(type(x_test), x_test.shape, x_test.dtype)
    print(type(y_test), y_test.shape, y_test.dtype)
    print(f"First training sample label: {y_train[0]}")

    # Reshape and normalize input data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 28 * 28))
    x_test = scaler.transform(x_test.reshape(-1, 28 * 28))
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    print(type(x_train), x_train.shape, x_train.dtype)
    print(type(y_train), y_train.shape, y_train.dtype)
    print(type(x_test), x_test.shape, x_test.dtype)
    print(type(y_test), y_test.shape, y_test.dtype)
    print(f"First training sample one-hot label: {y_train[0]}")

    return x_train, y_train, x_test, y_test


def main() -> None:
    """ Main Function """
    # Load MNIST dataset
    x_train, y_train, x_test, y_test = mnist_loader()

    # Set the model
    model = TwoLayersNN(input_size=x_train.shape[1], hidden_units=64, output_size=y_train.shape[1])

    # Set training parameters
    epochs = 10
    batch_size = 128
    alpha: float = 0.05

    losses: list[float] = []
    acc_train: list[float] = []
    acc_test: list[float] = []

    # Train the model
    for i in range(epochs):
        epoch_start = perf_counter()
        print(f"Epoch {i + 1}/{epochs}")
        # Generate random indices for mini-batch sampling to ensure every sample is used once per epoch
        indices = np_random.permutation(x_train.shape[0])
        for j in range(0, x_train.shape[0], batch_size):
            batch_start = perf_counter()
            # Get the mini-batch data
            batch_indices = indices[j:j + batch_size]
            x_train_batch = x_train[batch_indices]
            y_train_batch = y_train[batch_indices]

            # Use the method of gradient descent to update weights and biases
            grad = model.grad_descent(x_train_batch, y_train_batch)
            # Update weights and biases using gradient descent
            params = model.getter()
            for key in params.keys():
                params[key] -= alpha * grad[key]

            # Calculate loss and accuracy for training and test sets
            loss = model.loss(x_train_batch, y_train_batch)
            losses.append(loss)
            train_acc = model.accuracy(y_train, model.forward(x_train))
            test_acc = model.accuracy(y_test, model.forward(x_test))
            acc_train.append(train_acc)
            acc_test.append(test_acc)
            print(
                f"Batch {ceil(j / batch_size) + 1}/{ceil(x_train.shape[0] / batch_size)}\n"
                f"- Loss: {loss:.4f} "
                f"- Train Acc: {train_acc:.4f} "
                f"- Test Acc: {test_acc:.4f}"
            )
            batch_end = perf_counter()
            print(f"- Batch Time: {batch_end - batch_start:.4f} seconds")
        epoch_end = perf_counter()
        print(f"Epoch {i + 1} completed in {epoch_end - epoch_start:.4f} seconds")


if __name__ == "__main__":
    main()
