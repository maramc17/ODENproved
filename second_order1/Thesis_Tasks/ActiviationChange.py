import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import set_matplotlib_formats
from prettytable import PrettyTable

# Mara classes
from second_order1.classes.ODEsolverf import ODEsolverf
from second_order1.classes.Dictionary import Dictionary
from second_order1.classes.LossPlot import LossPlot

set_matplotlib_formats('pdf', 'svg')

plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Random seed initialization
seed = 0
# np.random.seed(seed)
tf.random.set_seed(seed)
# tensorFlow accuracy
tf.keras.backend.set_floatx('float64')
# Custom plot fontsize
import os

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin/'

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"

i = 0
D = Dictionary()
Dict = D.Dict
NString = ['Sigmoid', 'Tanh', 'Hard Sigmoid', 'Softmax']
DataSave = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

loss_plot = LossPlot()

for activation in [Dict["activation"]["relu"], Dict["activation"]["sigmoid"], Dict["activation"]["tanh"],
                   Dict["activation"]["elu"], Dict["activation"]["hard sigmoid"],
                   Dict["activation"]["linear"], Dict["activation"]["selu"], Dict["activation"]["softmax"]]:
    a, b, h, alpha, N = 0, 1, np.sqrt(2), 1, 100
    epochs = 25000

    order = 2
    diffEqf = "first"
    x = np.linspace(a, b, N, endpoint=False)[1:]  # training data: for x values without considering the end points
    architecture = [16]  # one hidden layer with 16 neurons
    initializer = Dict["initializer"]["GlorotNormal"]
    # activation = Dict["activation"]["sigmoid"]
    optimizer = Dict["optimizer"]["Adamax"]
    prediction_save = False

    weights_save = False

    solver = ODEsolverf(order, diffEqf, x, epochs, architecture, initializer, activation, optimizer, prediction_save,
                        weights_save, h, alpha)
    history = solver.train()
    epoch, loss = solver.get_loss(history)
    x_predict = np.linspace(a, b, num=N)  # testing data: will include the end points
    y_predict = solver.predict(x_predict)

    if activation == (Dict["activation"]["relu"]):
        relu = solver.predict(DataSave)
    elif activation == Dict["activation"]["sigmoid"]:
        loss_plot.add_plot_data(loss, N)
        sigmoid = solver.predict(DataSave)
    elif activation == Dict["activation"]["tanh"]:
        loss_plot.add_plot_data(loss, N)
        tanh = solver.predict(DataSave)
    elif activation == Dict["activation"]["elu"]:
        elu = solver.predict(DataSave)
    elif activation == Dict["activation"]["hard sigmoid"]:
        loss_plot.add_plot_data(loss, N)
        hardsigmoid = solver.predict(DataSave)
    elif activation == Dict["activation"]["linear"]:
        linear = solver.predict(DataSave)
    elif activation == Dict["activation"]["selu"]:
        selu = solver.predict(DataSave)
    else:
        loss_plot.add_plot_data(loss, N)
        softmax = solver.predict(DataSave)

loss_plot.plot_graph(NString)
table = PrettyTable(['x PTS', 'Relu', 'Sigmoid', 'Tanh', 'Elu', 'Hard Sigmoid', 'linear', 'Selu', 'Softmax'])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row(
        [round(DataSave[i], 1), round(*relu[i], 8), round(*sigmoid[i], 8), round(*tanh[i], 8), round(*elu[i], 8),
         round(*hardsigmoid[i], 8), round(*linear[i], 8), round(*selu[i], 8), round(*softmax[i], 8)])
print(table)

# x = ['1/10', '1/50', '1/100', '1/150', '1/200', '1/500']

# error = Error(EXACT, [PTS_10, PTS_50, PTS_100, PTS_150, PTS_200, PTS_500], x)
