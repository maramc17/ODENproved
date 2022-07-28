import matplotlib.pyplot as plt


def computeError(x, y):
    """
    This function does one job - it computes error given two lists. It can be broken down like so

    This -> [abs(exact - value) ** 2 for exact, value in zip(x, y)]
        creates a new list, each of whose elements are the squared absolute value of (exact - value)
        Exact being an element from the list of exact values taken at N=10000
        Value being the corresponding element from the list of test values taken between N=10...500
    This -> sum()
        iterates through all the values in a list, and returns their summation
    Thus, this function creates and returns the summation of the squared absolute value of the difference between
        a list of exact values and a list of test values
    """
    return sum([abs(exact - value) ** 2 for exact, value in zip(x, y)])


class Error:
    def __init__(self, y_exact, data, x_labels):
        """ METHOD: INIT
        VALUES
         x_labels: string       - a list containing the labels for the x-axis
         data_vals: number      - a list of values returned by solver.predict for N=10...500
         error_vals: number     - an empty list that will be filled with error values for N=10...500
         y_exact: number        - a list of values from solver.predict taken at n=1000
        """
        self.x_labels = x_labels
        self.data_vals = data
        self.error_vals = []
        self.y_exact = y_exact
        self.find_error()
        self.plot_error()

    def find_error(self):
        """
        METHOD: FIND ERROR
        PURPOSE:
        This function iterates through data_vals, which is a list of lists containing the output of solver.predict for
        each value N=1...500
        On each iteration, data is fed to computeError(), which does what its name implies, and the output is appended
            to self.error_vals
        """
        i = 0
        for data, x_axis in zip(self.data_vals, self.x_labels):
            print(x_axis)
            self.error_vals.append(computeError(self.y_exact, data))
            print(self.error_vals[i])
            i += 1

    def plot_error(self):
        """
        METHOD: PLOT ERROR
        PURPOSE:
            After find_error has done its job, we will have lists of x and y values concerning the error of our program
            This function creates a plot of those lists.
        """
        fig1 = plt.figure(1)
        ax = fig1.add_subplot()
        ax.plot(self.x_labels, self.error_vals, linewidth=2)
        ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.xlabel("h", fontsize=12)
        plt.ylabel("Error", fontsize=12)
        plt.tight_layout()
        plt.show()
