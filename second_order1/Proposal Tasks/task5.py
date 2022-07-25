import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf', 'svg')
import matplotlib

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
import time
import colorama
from matplotlib import animation

# Random seed initialization
seed = 0
# np.random.seed(seed)
tf.random.set_seed(seed)
# tensorFlow accuracy
tf.keras.backend.set_floatx('float64')
# Custom plot fontsize
import os

os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin/'
from matplotlib import cm
from matplotlib import rc

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
from matplotlib.colors import LogNorm


class Dictionary():

    def __init__(self):
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

        self.Dict_optimizers = {"Adadelta": tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07),
                                "Adagrad": tf.keras.optimizers.Adagrad(learning_rate=0.001,
                                                                       initial_accumulator_value=0.1, epsilon=1e-07),
                                "Adam": tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                                                 epsilon=1e-07, amsgrad=False),
                                "Adamax": tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                                                     epsilon=1e-07),
                                "Nadam": tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                                                   epsilon=1e-07),
                                "RMSprop": tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0,
                                                                       epsilon=1e-07, centered=False),
                                "SGD": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)}

        self.Dict = {"initializer": self.Dict_initializers, "activation": self.Dict_activations,
                     "optimizer": self.Dict_optimizers}


class DiffEqf():
    """
     w" + (1/t)*w' + (ha^2)(1- (w/1- aw));  ha = 1, a=1.
     Bc's:  w'(0) = 0 and w(1) = 0
    """

    def __init__(self, diffeqf, x, y, dydx, d2ydx2, alpha, h):
        """
          diffeq : name of the differential equation used (ex: diffeq = "first  ode")
          """
        self.diffeqf = diffeqf
        self.x = x
        self.y = y
        self.dydx = dydx
        self.d2ydx2 = d2ydx2

        if self.diffeqf == "first":
            self.eqf = self.d2ydx2 + (1 / self.x) * self.dydx + (1 - (self.y / 1 - self.y))


class ODEsolverf():

    def __init__(self, order, diffeqf, x, epochs, architecture, initializer, activation, optimizer, prediction_save,
                 weights_save, h, alpha):

        colorama.init()
        self.GREEN = colorama.Fore.GREEN
        self.RESET = colorama.Fore.RESET
        tf.keras.backend.set_floatx('float64')
        self.order = order
        self.diffeqf = diffeqf
        self.x = x
        self.n = len(self.x)
        self.epochs = epochs
        self.architecture = architecture
        self.initializer = initializer
        self.activation = activation
        self.optimizer = optimizer
        self.neural_net = self.build_model()  # self.neural_net_model(show = True)
        self.neural_net.summary()

        self.prediction_save = prediction_save
        self.weights_save = weights_save

        # Compile the model
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        self.neural_net.compile(loss=self.custom_cost(x), optimizer=self.optimizer, experimental_run_tf_function=False)
        # Calling tf.config.experimental_run_functions_eagerly(True) will make all invocations of tf.function run
        # eagerly instead of running as a traced graph function.See tf.config.run_functions_eagerly for an example.

        print("------- Model compiled -------")

        # Raise an exception is both prediction_save and weights_save are True
        if prediction_save and weights_save:
            raise Exception("Both prediciton_save and weights_save are set to True.")
        if prediction_save:
            self.predictions = []
        if weights_save:
            self.weights = []

    def build_model(self):
        """
        Builds a customized neural network model.
        """
        architecture = self.architecture
        initializer = self.initializer
        activation = self.activation

        nb_hidden_layers = len(architecture)
        input_tensor = tf.keras.layers.Input(shape=(1,))
        hidden_layers = []

        if nb_hidden_layers >= 1:
            hidden_layer = tf.keras.layers.Dense(architecture[0], kernel_initializer=initializer,
                                                 bias_initializer='zeros', activation=activation)(input_tensor)
            hidden_layers.append(hidden_layer)
            for i in range(1, nb_hidden_layers):
                hidden_layer = tf.keras.layers.Dense(architecture[i], kernel_initializer=initializer,
                                                     bias_initializer='zeros', activation=activation)(
                    hidden_layers[i - 1])
                hidden_layers.append(hidden_layer)
            output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer='zeros',
                                                 activation=tf.identity)(hidden_layers[-1])
        else:
            output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer, bias_initializer='zeros',
                                                 activation=tf.identity)(input_tensor)

        model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
        return model

    @tf.function
    def NN_output(self, x):
        """
        x : must be of shape = (?, 1)
        Returns the output of the neural net
        """
        y = self.neural_net(x)
        return y

    def y_gradients(self, x):
        """
        Computes the gradient of y.
        """
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y = self.NN_output(x)
            dy_dx = tape2.gradient(y, x)
        d2y_dx2 = tape1.gradient(dy_dx, x)
        return y, dy_dx, d2y_dx2

    def differential_cost(self, x, h, alpha):
        """
        Defines the differential cost function for one neural network
        input.
        """
        y, dydx, d2ydx2 = self.y_gradients(x)
        de = DiffEqf(self.diffeqf, x, y, dydx, d2ydx2, alpha, h)
        differential_equation = de.eqf
        return tf.square(differential_equation)

    def custom_cost(self, x):
        """
        Defines the cost function for a batch.
        """

        if self.order == 2:
            x_0 = 0
            x_1 = 1

            # y_prime = self.NN_output(np.asarray(x))[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]
            def loss(y_true, y_pred):
                differential_cost_term = tf.math.reduce_sum(self.differential_cost(x, h, alpha))
                boundary_cost_term = tf.square(self.NN_output(np.asarray([[x_1]]))[0][0] - 0)
                boundary_cost_term += tf.square(
                    (self.NN_output(x)[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]) - 0)
                return differential_cost_term / self.n + boundary_cost_term

            return loss

    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to x.
        """
        x = self.x
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        neural_net = self.neural_net

        # Train and save the predicitons
        if self.prediction_save:
            predictions = self.predictions

            # Define custom callback for predictions during training
            class PredictionCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    y_predict = neural_net.predict(x)
                    predictions.append(y_predict)
                    print('Prediction saved at epoch: {}'.format(epoch))

            start_time = time.time()
            history = neural_net.fit(x=x, y=x, batch_size=self.n, epochs=self.epochs, callbacks=[PredictionCallback()])
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")
            predictions = tf.reshape(predictions, (self.epochs, self.n))

        # Train and save the weights
        if self.weights_save:
            weights = self.weights

            # Define custom callback for weights during training
            class PredictionCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, log={}):
                    modelWeights = []
                    for i in range(1, len(neural_net.layers)):
                        layer_weights = neural_net.layers[i].get_weights()[0]
                        layer_biases = neural_net.layers[i].get_weights()[1]
                        modelWeights.append(layer_weights)
                        modelWeights.append(layer_biases)
                    weights.append(modelWeights)
                    print('Weights and biases saved at epoch: {}'.format(epoch))

            start_time = time.time()
            history = neural_net.fit(x=x, y=x, batch_size=self.n, epochs=self.epochs, callbacks=[PredictionCallback()])
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")

        # Train without any saving
        elif not self.prediction_save and not self.weights_save:
            start_time = time.time()
            history = neural_net.fit(x=x, y=x, batch_size=self.n, epochs=self.epochs)
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")

        return history

    def get_loss(self, history):
        """
        history : history of the training procedure returned by self.train
        Returns epochs and loss
        """
        epochs = history.epoch
        loss = history.history["loss"]
        return epochs, loss

    def predict(self, x_predict):
        """
        x_predict : domain of prediction (ex: x_predict = np.linspace(0, 1, 100))
        """
        domain_length = len(x_predict)
        x_predict = tf.convert_to_tensor(x_predict)
        x = tf.reshape(x_predict, (domain_length, 1))
        y_predict = self.neural_net.predict(x_predict)
        return y_predict

    def get_predictions(self):
        """
        Returns the neural net predictions at each epoch
        """
        if not self.prediction_save:
            raise Exception("The predictions have not been saved.")
        else:
            return self.predictions



D = Dictionary()
Dict = D.Dict

a, b, N, alpha, h = 0, 1, 100, 0.5, np.sqrt(0.5)
epochs = 20000

order = 2
diffEqf = "first"
x = np.linspace(a, b, N, endpoint=False)[1:]  # training data: for x values without considering the end points
architecture = [32]  # one hidden layer with 32 neurons
initializer = Dict["initializer"]["GlorotNormal"]
activation = Dict["activation"]["sigmoid"]
optimizer = Dict["optimizer"]["Adamax"]
prediction_save = False

weights_save = False

# Plotting for epochs
solver = ODEsolverf(order, diffEqf, x, epochs, architecture, initializer, activation, optimizer, prediction_save,
                    weights_save, h, alpha)
history = solver.train()
epoch, loss = solver.get_loss(history)
x_predict = np.linspace(a, b, N)  # testing data: will include the end points
y_predict = solver.predict(x_predict)
x_exact = np.linspace(a, b, N)
y_exact = 0.11374565155 * np.cos(1.5 * x_exact) + 0.11350373854 * np.sin(
        1.5 * x_exact) + 0.10788093628 + 0.0860238052 * x_exact + 0.047357124 * x_exact ** 2 + 0.0107362185 * x_exact ** 3


fig1 = plt.figure(1)
plt.plot(x_predict, y_predict, color="red", linewidth=2, label="Neural Network Solution")
plt.plot(x_exact, y_exact, color="blue", linewidth=2, linestyle='dotted', label="Exact Solution")
plt.xlabel("$x$", fontsize=15)
plt.ylabel("$y$", fontsize=15)

plt.legend(frameon=False)
plt.show()
fig1.savefig('figure1.png')
