import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt


class Error:

    def __init__(self, y_exact, y_predict):
        self.y_predict = y_predict
        self.y_exact = y_exact

    def find_error(self):

        error = 0
        for value, exact in zip(self.y_predict, self.y_exact):
            error += abs(exact - value)**2
        print(error)

        return error

    def plot_error(self, x, y):

        fig1 = plt.figure(1)
        ax = fig1.add_subplot()
        ax.plot(x, y, linewidth=2)
        ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.xlabel("h", fontsize=12)
        plt.ylabel("Error", fontsize=12)
        plt.tight_layout()

        plt.show()