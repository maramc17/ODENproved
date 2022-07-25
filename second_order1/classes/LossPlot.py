import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt



class LossPlot:

    def __init__(self, loss, NString):
        fig1 = plt.figure(1)
        ax = fig1.add_subplot()
        ax.plot(loss, linewidth=2, label=NString)
        ax.set_yscale('log')
        # plt.xlim(500, 25000)
        plt.ylim(10**(-6), 1)
        ax.legend(loc="upper right")
        #plt.legend(loc=5, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
