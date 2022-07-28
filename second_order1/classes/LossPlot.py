import matplotlib.pyplot as plt


def makeNString(N):
    return str(N) + " Data Points"


class LossPlot:
    def __init__(self):
        self.losses = []
        self.nums = []
        # self.loss = loss
        # self.NString = NString

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_num(self, num):
        self.nums.append(num)

    def add_plot_data(self, loss, num):
        self.add_loss(loss)
        self.add_num(num)

    def plot_graph(self, Nstring):
        fig1 = plt.figure(1)
        ax = fig1.add_subplot()
        for loss, num, name in zip(self.losses, self.nums, Nstring):
            ax.plot(loss, linewidth=2, label=name)
            ax.set_yscale('log')
            # plt.xlim(500, 25000)
            plt.ylim(10 ** (-6), 1)
            ax.legend(loc="upper right")
            # plt.legend(loc=5, bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.show()
