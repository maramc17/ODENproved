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
DataSave = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
for alpha in [1, 0.5, 0.1, 0.01, 0.001]:
    a, b, h, N = 0, 1, np.sqrt(2), 100
    NString = "α: " + str(alpha)
    epochs = 25000

    order = 2
    diffEqf = "first"
    x = np.linspace(a, b, N, endpoint=False)[1:]  # training data: for x values without considering the end points
    architecture = [16]  # one hidden layer with 16 neurons
    initializer = Dict["initializer"]["GlorotNormal"]
    activation = Dict["activation"]["sigmoid"]
    optimizer = Dict["optimizer"]["Adamax"]
    prediction_save = False

    weights_save = False

    solver = ODEsolverf(order, diffEqf, x, epochs, architecture, initializer, activation, optimizer, prediction_save,
                        weights_save, h, alpha)
    history = solver.train()
    epoch, loss = solver.get_loss(history)
    x_predict = np.linspace(a, b, num=N)  # testing data: will include the end points
    y_predict = solver.predict(x_predict)

    LossPlot(loss, NString)

    if alpha == 1:
        alpha1 = solver.predict(DataSave)
    elif alpha == 0.5:
        alpha2 = solver.predict(DataSave)
    elif alpha == 0.1:
        alpha3 = solver.predict(DataSave)
    elif alpha == 0.01:
        alpha4 = solver.predict(DataSave)
    else:
        alpha5 = solver.predict(DataSave)

table = PrettyTable(['x PTS', 'α: 1', 'α: 0.5', 'α: 0.1', 'α: 0.01', 'α: 0.001'])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row([round(DataSave[i], 1), round(*alpha1[i], 8), round(*alpha2[i], 8), round(*alpha3[i], 8),
                   round(*alpha4[i], 8), round(*alpha5[i], 8)])
print(table)

plt.show()
