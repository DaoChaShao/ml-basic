#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/9 15:21
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   10_grad_desc.py
# @Desc     :   

from pandas import DataFrame
from numpy import array
from PySide6.QtCore import Qt
from PySide6.QtCharts import QScatterSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QDoubleSpinBox)
from sys import argv, exit
from utils.D import gradient_descent


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._widget = QWidget(self)
        self._chart = QChart()
        self._view = QChartView(self._chart)
        self._btn_labels = ["Plot", "Clear", "Exit"]
        self._buttons = []
        self._spin_labels: list[str] = ["Alpha", "Epochs"]
        self._spins: dict[str, QDoubleSpinBox] = {}
        self._setup()
        self.setWindowTitle("Purely Mathematical Gradient Descent")
        self.resize(800, 400)

    def _setup(self):
        _layout = QVBoxLayout()
        _row = QHBoxLayout()

        _outer = QHBoxLayout()
        for lbl in self._spin_labels:
            inner = QHBoxLayout()
            label = QLabel(lbl, self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(label)

            spin = QDoubleSpinBox()
            if lbl == "Alpha":
                spin.setSingleStep(0.1)
                spin.setMinimum(0.01)
                spin.setMaximum(1.0)
                spin.setSingleStep(0.01)
                spin.setValue(0.01)
            elif lbl == "Epochs":
                spin.setMinimum(1)
                spin.setMaximum(10000)
                spin.setSingleStep(1)
                spin.setValue(100)
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(spin)

            _outer.addLayout(inner)
            self._spins[lbl] = spin
        _layout.addLayout(_outer)

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
        # Get parameters from spin boxes
        alpha = self._spins["Alpha"].value()
        epochs = int(self._spins["Epochs"].value())
        print(f"Alpha: {alpha}, Epochs: {epochs}")
        titles, dfs = data_setter(alpha, epochs, f)

        # Delete previous series
        self._chart.removeAllSeries()

        for i, line in enumerate(titles):
            series = QScatterSeries()
            series.setName(line)
            dataframe: DataFrame = dfs[i]
            # print(dataframe)
            print(type(dataframe), dataframe.shape)
            for j in range(dataframe.shape[0]):
                series.append(dataframe.at[j, "x"], dataframe.at[j, "y"])
            self._chart.addSeries(series)

        self._chart.setTitle(" & ".join(titles))
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


# Example function: f(x, y) = x^2 + y^2 + 0.1
def f(x):
    return x[0] ** 2 + x[1] ** 2 + 0.1


def data_setter(alpha: float, epochs: int, func) -> tuple[list[str], list[DataFrame]]:
    """ Set data for plotting
    :param alpha: learning rate
    :param epochs: number of epochs
    :param func: function to generate data
    :return: titles and dataframes
    """
    init_x = array([-0.2, 0.3])
    print(type(init_x), init_x.shape, init_x)

    x, f_vals = gradient_descent(func, init_x, lr=alpha, step_num=epochs)
    # print(f"After {epochs} epochs of Gradient Descent is {x}")
    # print(f"History of x during Gradient Descent:\n{f_vals}")
    print(type(f_vals), f_vals.shape)

    titles = ["Gradient Descent"]
    dfs = [DataFrame(f_vals, columns=["x", "y"])]
    return titles, dfs


def main() -> None:
    """ Main Function """
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())


if __name__ == "__main__":
    main()
