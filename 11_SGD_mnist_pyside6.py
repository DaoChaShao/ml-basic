#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/13 14:56
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   11_SGD_mnist_pyside6.py
# @Desc     :   

from PySide6.QtCore import Qt
from PySide6.QtCore import QThread, Signal
from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, )
from sys import argv, exit

from numpy import random as np_random, ceil
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.datasets import mnist
from time import perf_counter

from utils.network import Full2LayersNN


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


class Trainer(QThread):
    # loss, train_acc, test_acc
    updater = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self._x_train, self._y_train, self._x_test, self._y_test = mnist_loader()
        self._model = Full2LayersNN(
            input_size=self._x_train.shape[1],
            hidden_units=64,
            output_size=self._y_train.shape[1]
        )

        self._epochs: int = 5
        self._batch_size: int = 128
        self._alpha: float = 0.01

    def run(self):
        # Train the model
        for i in range(self._epochs):
            epoch_start = perf_counter()
            print(f"Epoch {i + 1}/{self._epochs}")
            # # shuffle per epoch (standard SGD)
            # - Generate random indices for mini-batch sampling to ensure every sample is used once per epoch
            indices = np_random.permutation(self._x_train.shape[0])
            for j in range(0, self._x_train.shape[0], self._batch_size):
                batch_start = perf_counter()
                # Get the mini-batch data
                batch_indices = indices[j:j + self._batch_size]
                x_train_batch = self._x_train[batch_indices]
                y_train_batch = self._y_train[batch_indices]

                # Use the method of gradient descent to update weights and biases
                grads = self._model.backward(x_train_batch, y_train_batch)

                # Update weights and biases using gradient descent
                params = self._model.getter()
                for key in params.keys():
                    params[key] -= self._alpha * grads[key]

                # Calculate loss and accuracy for training and test sets
                loss = self._model.loss(x_train_batch, y_train_batch)
                train_acc = self._model.accuracy(self._x_train, self._y_train)
                test_acc = self._model.accuracy(self._x_test, self._y_test)

                # Emit signal to update UI
                self.updater.emit(float(loss), float(train_acc), (test_acc))

                batch_end = perf_counter()
                print(f"Batch: {ceil(j / self._batch_size) + 1}: "
                      f"- loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f} "
                      f"[{batch_end - batch_start:.2f} sec]")
            epoch_end = perf_counter()
            print(f"Epoch {i + 1} completed in {epoch_end - epoch_start:.2f} sec")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mnist Train & Test Accuracy Visualization")
        self.resize(1200, 600)
        self._widget = QWidget(self)

        self._btn_labels = ["Train", "Clear", "Exit"]
        self._buttons = []

        self._data_loss, self._data_train, self._data_test = [], [], []
        self._series: dict[str, QLineSeries] = {}
        self._charts: dict[str, QChart] = {}
        self._views: dict[str, QChartView] = {}
        self._line_titles = ["Loss", "Train Acc", "Test Acc"]
        for title in self._line_titles:
            series = QLineSeries()
            chart = QChart()
            chart.addSeries(series)
            chart.legend().setVisible(False)
            view = QChartView(chart)
            self._charts[title] = chart
            self._series[title] = series
            self._views[title] = view

        self._trainer = Trainer()
        self._trainer.updater.connect(self.signal_updater)

        self._setup()

    def _setup(self):
        _layout = QVBoxLayout()
        _row_chart = QHBoxLayout()
        _row_btn = QHBoxLayout()

        # Chart Views
        for view in self._views.values():
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            _row_chart.addWidget(view)
        _layout.addLayout(_row_chart)

        funcs = [
            self._click2train,
            self._click2clear,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Clear":
                button.setEnabled(False)
            self._buttons.append(button)
            _row_btn.addWidget(button)
        _layout.addLayout(_row_btn)

        self._widget.setLayout(_layout)
        self.setCentralWidget(self._widget)

    def _click2train(self) -> None:
        """ Plot training and testing accuracy """
        self._trainer.start()

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Train":
                button.setEnabled(False)

    def _click2clear(self) -> None:
        """ Clear the chart """
        # Clear data lists
        for data in (self._data_loss, self._data_train, self._data_test):
            data.clear()
        # Clear series data
        for series in self._series.values():
            series.clear()
        # Reset charts
        for chart in self._charts.values():
            chart.createDefaultAxes()
            chart.setTitle("")

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Train":
                button.setEnabled(True)

    def signal_updater(self, loss: float, train_acc: float, test_acc: float) -> None:
        """ Slot to handle updates from the training thread
        :param loss: list of loss values
        :param train_acc: list of training accuracy values
        :param test_acc: list of testing accuracy values
        """
        self._data_loss.append(loss)
        self._data_train.append(train_acc)
        self._data_test.append(test_acc)

        # Update series with new data points
        for i, title in enumerate(self._line_titles):
            if title == "Loss":
                self._series[title].append(len(self._data_loss), self._data_loss[-1])
            elif title == "Train Acc":
                self._series[title].append(len(self._data_train), self._data_train[-1])
            elif title == "Test Acc":
                self._series[title].append(len(self._data_test), self._data_test[-1])

            # Update chart axes
            chart = self._charts[title]
            chart.createDefaultAxes()
            chart.setTitle(title)

            # Set axis titles and ranges
            axis_x = chart.axes(Qt.Orientation.Horizontal)[0]
            axis_y = chart.axes(Qt.Orientation.Vertical)[0]

            axis_x.setTitleText("Batches")
            axis_x.setRange(0, len(self._data_loss) + 1)

            if title == "Loss":
                axis_y.setTitleText("Loss")
                axis_y.setRange(0, max(self._data_loss) * 1.1)
            else:
                axis_y.setTitleText("Accuracy")
                axis_y.setRange(0, 1.0)


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
