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
for h in [2, np.sqrt(2), 1, np.sqrt(0.5)]:
    a, b, alpha, N = 0, 1, 1, 100
    NString = "h\u00b2: " + str(round(h**2, 1))
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

    if h == 2:
        h1 = solver.predict(DataSave)
    elif h == np.sqrt(2):
        h2 = solver.predict(DataSave)
    elif h == 1:
        h3 = solver.predict(DataSave)
    else:
        h4 = solver.predict(DataSave)

table = PrettyTable(['x PTS', "h\u00b2: 4", "h\u00b2: 2", "h\u00b2: 1", "h\u00b2: 0.5"])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row([round(DataSave[i], 1), round(*h1[i], 8), round(*h2[i], 8), round(*h3[i], 8),
                   round(*h4[i], 8)])
print(table)

plt.show()
