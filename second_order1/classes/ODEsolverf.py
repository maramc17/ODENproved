import colorama as colorama
import tensorflow as tf

from second_order1.classes.BuildModel import BuildModel


class ODEsolverf:
    def __init__(self, order, diffeqf, training_data, epochs, architecture, initializer, activation, optimizer,
                 prediction_save,
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
        buildModel = BuildModel(training_data, architecture, initializer, activation, optimizer, epochs)
        self.neural_net = BuildModel.get_model  # self.neural_net_model(show = True)

        self.prediction_save = prediction_save
        self.weights_save = weights_save

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

    def get_loss(self):
        """
        history : history of the training procedure returned by self.train
        Returns epochs and loss
        """
        epochs = self.neural_net.epoch
        loss = self.neural_net.history["loss"]
        return epochs, loss

    def predict(self, x_predict):
        """
        x_predict : domain of prediction (ex: x_predict = np.linespace(0, 1, 256))
        """
        domain_length = len(x_predict)
        x_predict = tf.convert_to_tensor(x_predict)
        x_predict = tf.reshape(x_predict, (domain_length, 1))
        y_predict = self.neural_net.predict(x=x_predict, batch_size=domain_length)
        return y_predict

    def get_predictions(self):
        """
        Returns the neural net predictions at each epoch
        """
        if not self.prediction_save:
            raise Exception("The predictions have not been saved.")
        else:
            return self.predictions
