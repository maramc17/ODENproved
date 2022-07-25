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
for architecture in [[4], [8], [16], [32], [64], [128]]:
    a, b, alpha, N, h = 0, 1, 1, 100, np.sqrt(2)
    NString = str(*architecture) + " Neurons"
    epochs = 25000

    order = 2
    diffEqf = "first"
    x = np.linspace(a, b, N, endpoint=False)[1:]  # training data: for x values without considering the end points
    # architecture = [16]  # one hidden layer with 16 neurons
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

    if architecture == [4]:
        N4 = solver.predict(DataSave)
    elif architecture == [8]:
        N8 = solver.predict(DataSave)
    elif architecture == [16]:
        N16 = solver.predict(DataSave)
    elif architecture == [32]:
        N32 = solver.predict(DataSave)
    elif architecture == [64]:
        N64 = solver.predict(DataSave)
    else:
        N128 = solver.predict(DataSave)
    i += 1

table = PrettyTable(['x PTS', "4 Neurons", "8 Neurons", "16 Neurons", "32 Neurons", "64 Neurons", "128 Neurons"])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row([round(DataSave[i], 1), round(*N4[i], 8), round(*N8[i], 8), round(*N16[i], 8),
                   round(*N32[i], 8), round(*N64[i], 8), round(*N128[i], 8)])
print(table)

plt.show()
