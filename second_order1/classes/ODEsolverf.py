import colorama as colorama
import tensorflow as tf
import numpy as np
import time
from second_order1.classes.DiffEqf import DiffEqf


class ODEsolverf:
    def __init__(self, order, diffeqf, training_data, epochs, architecture, initializer, activation, optimizer, prediction_save,
                 weights_save, h, alpha):

        colorama.init()
        self.GREEN = colorama.Fore.GREEN
        self.RESET = colorama.Fore.RESET
        tf.keras.backend.set_floatx('float64')
        self.order = order
        self.diffeqf = diffeqf
        self.training_data = training_data
        self.h = h
        self.alpha = alpha
        self.n = len(self.training_data)
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
        # training_data = self.training_data
        training_data = tf.convert_to_tensor(training_data)
        training_data = tf.transpose(training_data)
        self.neural_net.compile(loss=self.custom_cost(training_data, h), optimizer=self.optimizer, experimental_run_tf_function=False)
        # Calling tf.config.experimental_run_functions_eagerly(True) will make all invocations of tf.function run
        # eagerly instead of running as a traced graph function.See tf.config.run_functions_eagerly for an example.

        print("------- Model compiled -------")

        # Raise an exception if both prediction_save and weights_save are True
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
        training_data : must be of shape = (?, 1)
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

    def differential_cost(self, x, h):
        """
        Defines the differential cost function for one neural network
        input.
        """
        alpha = 1
        y, dydx, d2ydx2 = self.y_gradients(x)
        de = DiffEqf(self.diffeqf, x, y, dydx, d2ydx2, alpha, h)
        differential_equation = de.eqf
        return tf.square(differential_equation)

    def custom_cost(self, x, h):
        """
        Defines the cost function for a batch.
        """

        if self.order == 2:
            x_0 = 0
            x_1 = 1

            # y_prime = self.NN_output(np.asarray(training_data))[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]
            def loss(y_true, y_pred):
                differential_cost_term = tf.math.reduce_sum(self.differential_cost(x, h))
                boundary_cost_term = tf.square(self.NN_output(np.asarray([[x_1]]))[0][0] - 0)
                boundary_cost_term += tf.square(
                    (self.NN_output(x)[0][0] - self.NN_output(np.asarray([[x_0]]))[0][0]) - 0)
                return differential_cost_term / self.n + boundary_cost_term

            return loss

    def train(self):
        """
        neural_net : The built neural network returned by self.neural_net_model
        Trains the model according to training_data.
        """
        x = self.training_data
        x = tf.convert_to_tensor(x)
        x = tf.reshape(x, (self.n, 1))
        neural_net = self.neural_net

        # Train and save the predictions
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
        x_predict : domain of prediction (ex: x_predict = np.linespace(0, 1, 256))
        """
        domain_length = len(x_predict)
        x_predict_t = tf.convert_to_tensor(x_predict)
        x_predict = tf.reshape(x_predict_t, shape=(domain_length, 1))
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
