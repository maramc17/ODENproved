import tensorflow as tf
import time
import colorama as colorama
from second_order1.classes.DiffEqf import DiffEqf
import numpy as np


class BuildModel:
    def __init__(self, training_data, layers, init, active, opt, epochs):
        colorama.init()
        self.GREEN = colorama.Fore.GREEN
        self.RESET = colorama.Fore.RESET

        self.layers, self.epochs = layers, epochs
        self.initializer, self.activation, self.optimizer = init, active, opt
        self.training_data = tf.convert_to_tensor(training_data)
        self.N = len(self.training_data)
        self.training_data = tf.reshape(self.training_data, (self.N, 1))
        self.predictions = []
        self.weights = []

        self.model = self.build_model()


        print("------------- MODEL BUILD COMPLETE -----------------")

    @property
    def get_model(self):
        # Compile the model
        self.model.compile(loss=self.custom_cost(self.training_data, np.sqrt(2)), optimizer=self.optimizer)
        self.model.summary()
        return  self.train()

    def build_model(self):
        """
        Builds a customized neural network model.
        """
        layers = self.layers
        num_hlayers = len(layers)
        kernel_initializer = self.initializer
        activation = self.activation
        model = tf.keras.Sequential()

        if num_hlayers >= 1:
            """
            Build the input layer
            """
            model.add(tf.keras.layers.InputLayer(input_shape=(1,)))
            """
            Build the hidden nodes
            """
            for nodes in layers:
                model.add(tf.keras.layers.Dense(nodes,
                                                activation=activation,
                                                kernel_initializer=kernel_initializer
                                                ))
            """
            Build the output layer
            """
            model.add(tf.keras.layers.Dense(1,
                                            activation=activation,
                                            kernel_initializer=kernel_initializer
                                            ))
        else:
            model.add(tf.keras.layers.Dense(1, activation=activation,
                                            kernel_initializer=kernel_initializer
                                            ))


        return model

    @tf.function
    def NN_output(self, x):
        """
        x : must be of shape = (?, 1)
        Returns the output of the neural net
        """
        model = self.model
        y = model(x)
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

    def differential_cost(self, x, h):
        """
        Defines the differential cost function for one neural network
        input.
        """
        alpha = 1
        y, dydx, d2ydx2 = self.y_gradients(x)
        de = DiffEqf("first", x, y, dydx, d2ydx2, alpha, h)
        differential_equation = de.eqf
        return tf.square(differential_equation)

    def custom_cost(self, x, h):
        """
        Defines the cost function for a batch.
        """
        #if self.order == 2:
        x_0 = 0
        x_1 = 1

        # y_prime = self.NN_output(np.asarray(x))[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]
        def loss(y_true, y_pred):
            differential_cost_term = tf.math.reduce_sum(self.differential_cost(x, h))
            boundary_cost_term = tf.square(self.NN_output(np.asarray([[x_1]]))[0][0] - 0)
            boundary_cost_term += tf.square(
                (self.NN_output(x)[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]) - 0)
            return differential_cost_term / self.N + boundary_cost_term

        return loss

    def train_new(self, in_model):

        start_time = time.time()
        hist = in_model.fit(x=self.training_data, y=self.training_data, batch_size=32,
                            epochs=self.epochs)
        print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
        print(f"{self.RESET}")
        return hist

    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to x.
        """
        training_data = self.training_data
        neural_net = self.model

        # Train and save the predictions
        if True: #self.prediction_save:
            predictions = self.predictions

            # Define custom callback for predictions during training
            class PredictionCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    y_predict = neural_net.predict(training_data)
                    predictions.append(y_predict)
                    print('Prediction saved at epoch: {}'.format(epoch))

            start_time = time.time()
            history = neural_net.fit(x=training_data, y=training_data,
                                     batch_size=self.N, epochs=self.epochs, callbacks=[PredictionCallback()])
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")
            predictions = tf.reshape(predictions, (self.epochs, self.N))

        # Train and save the weights
        if True: # self.weights_save:
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
            history = neural_net.fit(x=training_data, y=training_data, batch_size=self.N, epochs=self.epochs,
                                     callbacks=[PredictionCallback()])
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")

        # Train without any saving
        elif not self.prediction_save and not self.weights_save:
            start_time = time.time()
            history = neural_net.fit(x=training_data, y=training_data, batch_size=self.N, epochs=self.epochs)
            print(f"{self.GREEN}---   %s seconds ---  " % (time.time() - start_time))
            print(f"{self.RESET}")

        return history
