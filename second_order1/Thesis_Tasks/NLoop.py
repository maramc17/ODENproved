import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import set_matplotlib_formats
from prettytable import PrettyTable

# Mara classes
from second_order1.classes.ODEsolverf import ODEsolverf
from second_order1.classes.Dictionary import Dictionary
from second_order1.classes.LossPlot import LossPlot
from second_order1.classes.Error import Error

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
for N in [10, 50, 100, 150, 200, 500, 1000]:
    a, b, h, alpha = 0, 1, np.sqrt(2), 1
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

    if N != 1000:
        NString = str(N) + " Data Points"
        LossPlot(loss, NString)
        match N:
            case 10:
                PTS_10 = solver.predict(DataSave)
            case 50:
                PTS_50 = solver.predict(DataSave)
            case 100:
                PTS_100 = solver.predict(DataSave)
            case 150:
                PTS_150 = solver.predict(DataSave)
            case 200:
                PTS_200 = solver.predict(DataSave)
            case 500:
                PTS_500 = solver.predict(DataSave)
    else:
        EXACT = solver.predict(DataSave)

table = PrettyTable(['X PTS', 'Y_E', 'Y_10', 'Y_50', 'Y_100', 'Y_150', 'Y_200', 'Y_500'])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row(
        [round(DataSave[i], 1), round(*EXACT[i], 8), round(*PTS_10[i], 8), round(*PTS_50[i], 8), round(*PTS_100[i], 8),
         round(*PTS_150[i], 8), round(*PTS_200[i], 8), round(*PTS_500[i], 8)])
print(table)

plt.show()

i = 0
y = []
x = ['1/10', '1/50', '1/100', '1/150', '1/200', '1/500']
for points in [PTS_10, PTS_50, PTS_100, PTS_150, PTS_200, PTS_500]:
    print(x[i])
    error = Error(EXACT, points)
    y.append(error.find_error())
    i += 1

error.plot_error(x, y)
