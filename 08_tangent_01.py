#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/8 15:39
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   08_tangent_01.py
# @Desc     :
# 1. Derivative the slope of the model point or points
# 2. Drae the tangent line with the slope
# 3. Judge the trend of the model with the tangent line

from pandas import DataFrame
from numpy import ndarray, arange
from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, )
from random import randint
from sys import argv, exit
from utils.D import central_diff, exact_gradient


def model(features: ndarray) -> ndarray:
    """ Example model that has been trained and optimized
    :param features: input feature, which can be called x also
    :return: output value
    """
    return 0.01 * features ** 2 + 0.1 * features


def tangent_func(f, point):
    """ Compute the tangent line with approximate slope of derivative function f at point x
    :param f: function to compute the tangent line for
    :param point: point at which to compute the tangent line
    :return: function representing the tangent line at point x
    """
    y = f(point)
    slope = central_diff(f, point)
    print(f"Slope at point {point:.2f} is approximately {slope:.4f}")
    # slope = exact_gradient(point)
    # print(f"Exact slope at point {point:.2f} is {slope:.4f}")
    # Tangent line equation: y = slope * x + intercept
    intercept = y - slope * point
    return lambda x: slope * x + intercept


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._widget = QWidget(self)
        self._chart = QChart()
        self._view = QChartView(self._chart)
        self._btn_labels = ["Plot", "Clear", "Exit"]
        self._buttons = []
        self._setup()
        self.setWindowTitle("Derivative & Tangent Visualization")
        self.resize(800, 400)

        self._line_titles: list[str] = []
        self._line_dfs: list[DataFrame] = []

    def _setup(self):
        _layout = QVBoxLayout()
        _row = QHBoxLayout()

        # Chart View
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _layout.addWidget(self._view)

        funcs = [
            self._click2plot,
            self._click2clear,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Clear":
                button.setEnabled(False)
            self._buttons.append(button)
            _row.addWidget(button)
        _layout.addLayout(_row)

        self._widget.setLayout(_layout)
        self.setCentralWidget(self._widget)

    def _click2plot(self) -> None:
        """ Plot random data points """
        # Delete previous series
        self._chart.removeAllSeries()

        for i, line in enumerate(self._line_titles):
            series = QLineSeries()
            series.setName(line)
            dataframe: DataFrame = self._line_dfs[i]
            # print(dataframe)
            print(type(dataframe), dataframe.shape)
            for j in range(dataframe.shape[0]):
                series.append(dataframe.at[j, "x"], dataframe.at[j, "y"])
            self._chart.addSeries(series)

        self._chart.setTitle(f"{self._line_titles[0]} & {self._line_titles[1]}")
        self._chart.createDefaultAxes()

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(False)

    def _click2clear(self) -> None:
        """ Clear the chart """
        self._chart.setTitle("")
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(True)

    def load_data(self, titles: list[str], dfs: list[DataFrame]) -> None:
        """ Load data for plotting
        :param titles: list of line titles
        :param dfs: list of DataFrames containing x and y values
        """
        self._line_titles = titles
        self._line_dfs = dfs


def main() -> None:
    """ Main Function """
    app = QApplication(argv)

    # Generate x values from 0 to 10 with step size 0.1
    features = arange(0, 20, 0.1)
    # Compute y values using the example function
    y_pred = model(features)

    # Pick a random point to compute the slope of a tangent line
    index = randint(0, len(features) - 1)
    point = features[index]

    # Get a tangent line function at the selected point
    tangent = tangent_func(model, point)
    # Draw the tangent line with the tangent function with the effective range of the model
    y_tangent = tangent(features)

    # Plot the tangent line and the model function with PySide6
    data_model: DataFrame = DataFrame({"x": features, "y": y_pred})
    data_tangent: DataFrame = DataFrame({"x": features, "y": y_tangent})
    line_titles: list[str] = ["Model", "Tangent"]
    line_dfs: list[DataFrame] = [data_model, data_tangent]

    # Create and show the main window
    window = MainWindow()
    window.load_data(line_titles, line_dfs)

    # Show the window and start the application event loop
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
