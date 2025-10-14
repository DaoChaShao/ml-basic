# python
# !/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/14 15:30
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   12_optimiser_mnist_pyside6.py
# @Desc     :

from PySide6.QtCore import Qt
from PySide6.QtCore import QThread, Signal
from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton)
from sys import argv, exit

from numpy import random as np_random, ceil
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.datasets import mnist
from time import perf_counter

from utils.network import Full2LayersNN
from utils.optimiser import SGD, Momentum, AdaGrad, RMSProp, Adam


def mnist_loader():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, 28 * 28))
    x_test = scaler.transform(x_test.reshape(-1, 28 * 28))
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))
    return x_train, y_train, x_test, y_test


class Trainer(QThread):
    # name, loss, train_acc, test_acc
    updater = Signal(str, float, float, float)

    def __init__(self, optimizers: dict[str, object], epochs: int = 5, batch_size: int = 128):
        super().__init__()
        self._x_train, self._y_train, self._x_test, self._y_test = mnist_loader()
        self._epochs = epochs
        self._batch_size = batch_size

        # Build a model for each optimizer
        self._optimizers = optimizers
        self._models: dict[str, Full2LayersNN] = {
            name: Full2LayersNN(
                input_size=self._x_train.shape[1],
                hidden_units=64,
                output_size=self._y_train.shape[1]
            )
            for name in self._optimizers.keys()
        }

    def run(self):
        n_samples = self._x_train.shape[0]
        for i in range(self._epochs):
            epoch_start = perf_counter()
            print(f"Epoch {i + 1}/{self._epochs}")
            indices = np_random.permutation(n_samples)

            for j in range(0, n_samples, self._batch_size):
                batch_start = perf_counter()
                batch_indices = indices[j:j + self._batch_size]
                xb = self._x_train[batch_indices]
                yb = self._y_train[batch_indices]

                # Update the weights and biases with different optimizer using the same model
                for name, model in self._models.items():
                    # Get the gradients via backpropagation
                    grads = model.backward(xb, yb)
                    # Get current parameters
                    params = model.getter()
                    # Update parameters using the specific optimizer
                    self._optimizers[name].update(params, grads)

                    loss = model.loss(xb, yb)
                    train_acc = model.accuracy(self._x_train, self._y_train)
                    test_acc = model.accuracy(self._x_test, self._y_test)

                    # Emit signal to update the main window
                    self.updater.emit(name, float(loss), float(train_acc), float(test_acc))

                batch_end = perf_counter()
                print(f"Batch: {ceil(j / self._batch_size) + 1}: "
                      f"[{batch_end - batch_start:.2f} sec]")

            epoch_end = perf_counter()
            print(f"Epoch {i + 1} completed in {epoch_end - epoch_start:.2f} sec")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST 多优化器训练对比")
        self.resize(1200, 600)
        self._widget = QWidget(self)

        self._btn_labels: list[str] = ["Train", "Clear", "Exit"]
        self._buttons: list[QPushButton] = []

        # Add different optimizers for comparing
        self._optimizers: dict[str, object] = {
            "SGD": SGD(alpha=0.01),
            "Momentum": Momentum(alpha=0.01, beta=0.9),
            "AdaGrad": AdaGrad(alpha=0.01),
            "RMSProp": RMSProp(alpha=0.001, beta=0.9),
            "Adam": Adam(alpha=0.001, beta1=0.9, beta2=0.999),
        }
        self._opt_names = list(self._optimizers.keys())

        # Set up 3 metrics to track
        self._metrics = ["Loss", "Train Acc", "Test Acc"]
        self._data: dict[str, dict[str, list[float]]] = {
            name: {m: [] for m in self._metrics} for name in self._opt_names
        }

        # Charts and series for visualization
        self._charts: dict[str, QChart] = {}
        self._views: dict[str, QChartView] = {}
        self._series: dict[str, dict[str, QLineSeries]] = {m: {} for m in self._metrics}

        for metric in self._metrics:
            chart = QChart()
            chart.setTitle(metric)
            chart.legend().setVisible(True)
            chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

            # Create a line series for each optimizer
            for name in self._opt_names:
                s = QLineSeries()
                s.setName(name)
                chart.addSeries(s)
                self._series[metric][name] = s

            chart.createDefaultAxes()
            view = QChartView(chart)
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._charts[metric] = chart
            self._views[metric] = view

        # Trainer thread
        self._trainer = Trainer(self._optimizers)
        self._trainer.updater.connect(self.signal_updater)

        self._setup()

    def _setup(self):
        layout = QVBoxLayout()
        row_chart = QHBoxLayout()
        row_btn = QHBoxLayout()

        for v in self._views.values():
            row_chart.addWidget(v)
        layout.addLayout(row_chart)

        funcs = [self._click2train, self._click2clear, self.close]
        for i, label in enumerate(self._btn_labels):
            btn = QPushButton(label, self)
            btn.clicked.connect(funcs[i])
            if label == "Clear":
                btn.setEnabled(False)
            self._buttons.append(btn)
            row_btn.addWidget(btn)
        layout.addLayout(row_btn)

        self._widget.setLayout(layout)
        self.setCentralWidget(self._widget)

    def _click2train(self) -> None:
        self._trainer.start()
        for b in self._buttons:
            if b.text() == "Clear":
                b.setEnabled(True)
            if b.text() == "Train":
                b.setEnabled(False)

    def _click2clear(self) -> None:
        # Clear all data and reset charts
        for name in self._opt_names:
            for m in self._metrics:
                self._data[name][m].clear()
        for metric in self._metrics:
            for s in self._series[metric].values():
                s.clear()
            self._charts[metric].createDefaultAxes()

        for b in self._buttons:
            if b.text() == "Clear":
                b.setEnabled(False)
            if b.text() == "Train":
                b.setEnabled(True)

    def signal_updater(self, name: str, loss: float, train_acc: float, test_acc: float) -> None:
        # Add new data points
        self._data[name]["Loss"].append(loss)
        self._data[name]["Train Acc"].append(train_acc)
        self._data[name]["Test Acc"].append(test_acc)

        # Append to series
        step = len(self._data[name]["Loss"])
        self._series["Loss"][name].append(step, loss)
        self._series["Train Acc"][name].append(step, train_acc)
        self._series["Test Acc"][name].append(step, test_acc)

        # Update axes
        self._update_axes()

    def _update_axes(self):
        # Update axes ranges based on current data at the same time
        # Find max steps and max loss across all optimizers
        max_steps = 1
        max_loss = 1e-6
        for name in self._opt_names:
            max_steps = max(max_steps, len(self._data[name]["Loss"]))
            if len(self._data[name]["Loss"]) > 0:
                max_loss = max(max_loss, max(self._data[name]["Loss"]))

        for metric in self._metrics:
            chart = self._charts[metric]
            axes_x = chart.axes(Qt.Orientation.Horizontal)
            axes_y = chart.axes(Qt.Orientation.Vertical)
            if not axes_x or not axes_y:
                chart.createDefaultAxes()
                axes_x = chart.axes(Qt.Orientation.Horizontal)
                axes_y = chart.axes(Qt.Orientation.Vertical)
                if not axes_x or not axes_y:
                    continue
            ax_x = axes_x[0]
            ax_y = axes_y[0]

            ax_x.setTitleText("Batches")
            ax_x.setRange(0, max_steps + 1)

            if metric == "Loss":
                ax_y.setTitleText("Loss")
                ax_y.setRange(0, max_loss * 1.1)
            else:
                ax_y.setTitleText("Accuracy")
                ax_y.setRange(0.0, 1.0)


def main() -> None:
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
