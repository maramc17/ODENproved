import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from IPython.display import set_matplotlib_formats
# import matplotlib_inline
from prettytable import PrettyTable

# Mara classes
from second_order1.classes.ODEsolverf import ODEsolverf
from second_order1.classes.Dictionary import Dictionary
from second_order1.classes.LossPlot import LossPlot
from second_order1.classes.Error import Error

# set_matplotlib_formats('pdf', 'svg')

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
Nstring = ["10 Data points", "50 Data points", "100 Data points", "150 Data points",
           "200 Data points", "500 Data points"]

loss_plot = LossPlot()
architecture = [16]  # one hidden layer with 16 neurons
initializer = Dict["initializer"]["GlorotNormal"]
activation = Dict["activation"]["sigmoid"]
optimizer = Dict["optimizer"]["Adamax"]
a, b, h, alpha = 0, 1, np.sqrt(2), 1
epochs = 250
order = 2
diffEqf = "first"
prediction_save = False
weights_save = False

models = []
numbers = [10, 50, 100, 150, 200, 500, 1000]
for N in numbers:
    training_data = np.linspace(a, b, N, endpoint=False)[1:]  # for training values without considering the end points

    solver = ODEsolverf(order, diffEqf, training_data, epochs, architecture, initializer, activation, optimizer,
                        prediction_save,
                        weights_save, h, alpha)

    epoch, loss = solver.get_loss()
    loss_plot.add_plot_data(loss, N)
    models.append(solver)
    x_predict = np.linspace(a, b, num=N)  # testing data: will include the end points
    # y_predict = solver.predict(x_predict)

PTS = []
for model, number in zip(models, numbers):
    PTS.append(model.predict(DataSave))

loss_plot.plot_graph(Nstring)
table = PrettyTable(['X PTS', 'Y_E', 'Y_10', 'Y_50', 'Y_100', 'Y_150', 'Y_200', 'Y_500'])

for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    table.add_row(
        [round(DataSave[i], 1), round(*PTS[6][i], 8), round(*PTS[0][i], 8), round(*PTS[1][i], 8), round(*PTS[2][i], 8),
         round(*PTS[3][i], 8), round(*PTS[4][i], 8), round(*PTS[5][i], 8)])
print(table)

x = ['1/10', '1/50', '1/100', '1/150', '1/200', '1/500']

error = Error(EXACT, [PTS_10, PTS_50, PTS_100, PTS_150, PTS_200, PTS_500], x)
