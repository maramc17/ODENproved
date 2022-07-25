import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import set_matplotlib_formats
from prettytable import PrettyTable

# Mara classes
from second_order1.classes.ODEsolverf import ODEsolverf
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

class Dictionary:

    def __init__(self, LR):
        self.Dict_initializers = {"GlorotNormal": "GlorotNormal",
                                  "GlorotUniform": "GlorotUniform",
                                  "Ones": "Ones",
                                  "RandomNormal": "RandomNormal",
                                  "RandomUniform": "RandomUniform",
                                  "Zeros": "Zeros"}

        self.Dict_activations = {"relu": tf.nn.relu,
                                 "sigmoid": tf.nn.sigmoid,
                                 "tanh": tf.nn.tanh,
                                 "elu": tf.nn.elu,
                                 "hard sigmoid": tf.keras.activations.hard_sigmoid,
                                 "linear": tf.keras.activations.linear,
                                 "selu": tf.keras.activations.selu,
                                 "softmax": tf.keras.activations.softmax}

        self.Dict_optimizers = {"Adadelta": tf.keras.optimizers.Adadelta(learning_rate=LR, rho=0.95, epsilon=1e-07),
                                "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=LR,
                                                                       initial_accumulator_value=0.1, epsilon=1e-07),
                                "Adam": tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999,
                                                                 epsilon=1e-07, amsgrad=False),
                                "Adamax": tf.keras.optimizers.Adamax(learning_rate=LR, beta_1=0.9, beta_2=0.999,
                                                                     epsilon=1e-07),
                                "Nadam": tf.keras.optimizers.Nadam(learning_rate=LR, beta_1=0.9, beta_2=0.999,
                                                                   epsilon=1e-07),
                                "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=LR, rho=0.9, momentum=0.0,
                                                                       epsilon=1e-07, centered=False),
                                "SGD": tf.keras.optimizers.SGD(learning_rate=LR, momentum=0.0, nesterov=False)}

        self.Dict = {"initializer": self.Dict_initializers, "activation": self.Dict_activations,
                     "optimizer": self.Dict_optimizers}


DataSave = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
for LR in [0.1, 0.01, 0.001, 0.0001]:
    D = Dictionary(LR)
    Dict = D.Dict
    a, b, h, alpha, N = 0, 1, np.sqrt(2), 1, 100
    NString = "Learning Rate: " + str(LR)
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
    epoch, loss = solver.get_loss(history)  # LOSS???
    x_predict = np.linspace(a, b, num=N)  # testing data: will include the end points
    y_predict = solver.predict(x_predict)

    LossPlot(loss, NString)

    if LR == 0.1:
        LR1 = solver.predict(DataSave)
    elif LR == 0.01:
        LR2 = solver.predict(DataSave)
    elif LR == 0.001:
        LR3 = solver.predict(DataSave)
    else:
        LR4 = solver.predict(DataSave)

table = PrettyTable(['x PTS', 'Learning Rate: 0.1', 'Learning Rate: 0.01', 'Learning Rate: 0.001',
                     'Learning Rate: 0.0001'])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row([round(DataSave[i], 1), round(*LR1[i], 8), round(*LR2[i], 8), round(*LR3[i], 8),
                   round(*LR4[i], 8)])
print(table)

plt.show()