"""
A network definition

@date: 15/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from keras.models import Sequential
from keras import optimizers
from useful.functions import nmap, plot_color, plot_3d, history_plot, white_space, title
from useful.data import Data


class Network:
    def __init__(self, data: Data):
        self.model = Sequential()
        self.data = data
        self.trained = False
        self.history = {}
        self.validation_set = True

    @white_space
    def build(self, loss_function: callable = 'mean_squared_error'):
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=loss_function, optimizer=sgd, metrics=['accuracy'])
        self.model.summary()

    @white_space
    def train(self,
              split_ratio: float = 0.5,
              validate: bool = True,
              plot_history: bool = False,
              plot_acc: bool = False,
              plot_loss: bool = False,
              save_link: str = ""
              ) -> None:
        """
        Split data, train the model, print weights, and graph the learning
        :param split_ratio:
        :param validate: use validation set ?
        :param plot_history: plot accuracy and loss ?
        :param plot_acc: plot accuracy ?
        :param plot_loss: plot loss ?
        :param save_link: save the plots ?
        """
        self.trained = True
        self.validation_set = validate
        self.data.split(split_ratio)

        print("Learning set : {} values".format(len(self.data.question_training)))
        print("Training set : {} values".format(len(self.data.question_testing)))

        if self.validation_set:
            history = self.model.fit(self.data.question_training,
                                     self.data.expected_training,
                                     epochs=50,
                                     validation_data=(
                                         self.data.question_testing,
                                         self.data.expected_testing))
        else:
            history = self.model.fit(self.data.question_training,
                                     self.data.expected_training,
                                     epochs=50)
        self.history = history.history
        self.print_weights()

        if plot_history or plot_acc:
            self.graph_history('acc')
        if plot_history or plot_loss:
            self.graph_history('loss')
        if save_link:
            self.graph_history('acc', save_link)
            self.graph_history('loss', save_link)

    def print_weights(self):
        title(" Weights : ")
        for w in self.model.get_weights()[0]:
            print("->", w[0])

    @property
    def predictions(self):
        return [self.predict(e) for e in self.data.question_data]

    def predict(self, inp):
        if self.trained is False:
            self.train()
        return self.model.predict(np.array([inp]))[0][0]

    def __call__(self, inp):
        return self.predict(inp)

    def graph_color(self, save_link: str = "") -> None:
        """
        plot a 2D colored graph of a 2D array
        :param save_link: if you want to save the plot
        """
        t = np.arange(20)/200
        x = np.concatenate((t-max(t), t))
        xy = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
        z_exp = np.split(nmap(self.data.func, xy), len(x))

        plt1 = plot_color(z_exp, x)
        z_val = np.split(nmap(self.predict, xy), len(x))
        plt2 = plot_color(z_val, x)

        if save_link is not "":
            plt1.savefig(save_link + "/expected.png")
            plt2.savefig(save_link + "/result.png")
        else:
            plt1.show()
            plt2.show()

    def graph3d(self, save_link: str = "") -> None:
        """
        plot expected and result value over 2 axis
        :param save_link: if you want to save the plot
        """
        x = self.data.question_data[:, 0]
        y = self.data.question_data[:, 1]
        z_exp = self.data.expected_data
        z_val = [self(val) for val in self.data.question_data]

        plt = plot_3d(x, y, z_exp, z_val)
        if save_link is not "":
            plt.savefig(save_link + "/3D_plot.png")
        else:
            plt.show()

    def graph_history(self, key: str, save_link: str = "") -> None:
        """
        plot the learning history
        """
        if self.history:
            plt = history_plot(self.history, key, self.validation_set)
            if save_link:
                plt.savefig(save_link + "/" + key + ".png")
            else:
                plt.show()
