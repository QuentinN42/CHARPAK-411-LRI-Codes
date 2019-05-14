"""
Usefull thinks...

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from keras.models import Sequential
from keras import optimizers
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def nmap(func, l):
    return np.array(list(map(func, l)))


def f(e):
    return int(e*100)/100


def shuffle(tab: np.array) -> np.array:
    indices = np.arange(len(tab))
    np.random.shuffle(indices)
    return tab[indices]


def generate(n: int = 100, div: int = 100) -> np.array:
    t = np.arange(n) / div
    tab = np.concatenate((t-max(t), t))
    print(min(tab), max(tab))
    return np.transpose([np.tile(tab, len(tab)), np.repeat(tab, len(tab))])


def moy(tab: np.array) -> float:
    return sum(tab)/len(tab)


def plot_color(z: np.array, x: np.array = None, y: np.array = None, nb_ticks: int = 5) -> plt:
    fig, ax = plt.subplots()
    im = ax.imshow(z)
    plt.plot([len(z)-.5, -.5], [-.5, len(z)-.5], '-k')
    fig.tight_layout()
    plt.colorbar(im)
    if x is not None:
        if y is None:
            y = x
        ticks_pos_x = nmap(int, np.arange(nb_ticks)*(len(x)-1)/(nb_ticks-1))
        ticks_pos_y = nmap(int, np.arange(nb_ticks) * (len(y) - 1) / (nb_ticks - 1))
        ax.set_xticks(ticks_pos_x)
        ax.set_yticks(ticks_pos_y)
        ax.set_xticklabels([x[i] for i in ticks_pos_x])
        ax.set_yticklabels([y[i] for i in ticks_pos_y])
    plt.show()
    return plt


class LearningData:
    def __init__(self, tab: np.array):
        self.data = tab
        self.training = np.array([])
        self.testing = np.array([])

    def __len__(self):
        return len(self.data)

    def split(self, index: int) -> None:
        self.training, self.testing = self.data[:index], self.data[index:]


class Data:
    def __init__(self, tab: np.array, func):
        self.func = func
        if len(tab) == 0:
            raise EnvironmentError('Data must have some data')
        self.raw_data = tab
        self.question = LearningData(shuffle(self.raw_data))
        self.expected = LearningData(np.array([self.func(e) for e in self.question.data]))

    def __len__(self):
        return len(self.question)

    def split(self, learning_set_ratio: float = 0.5) -> None:
        split_at = int(learning_set_ratio * len(self))
        self.question.split(split_at)
        self.expected.split(split_at)


class Network:
    def __init__(self, data: Data):
        self.model = Sequential()
        self.data = data
        self.trained = False

    def build(self):
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        self.model.summary()

    def train(self, split_ratio: float = 0.8):
        self.trained = True
        self.data.split(split_ratio)
        self.model.fit(self.data.question.training,
                       self.data.expected.training,
                       epochs=1, validation_data=(
                        self.data.question.testing,
                        self.data.expected.testing))
        print("="*20 + " Weights : " + "="*20)
        for wts in self.model.get_weights():
            print(" | ".join([str(w) for w in wts]))

    @property
    def predictions(self):
        return [self.predict(e) for e in self.data.question.data]

    def predict(self, inp):
        if self.trained is False:
            self.train()
        return self.model.predict(np.array([inp]))[0][0]

    def __call__(self, inp):
        return self.predict(inp)

    def graph(self):
        exp = sorted(self.data.expected.data)
        questions = sorted(self.data.question.data, key=self.data.func)
        res = [self.predict(q) for q in questions]
        plt.plot(exp, '-r', label='expected')
        plt.plot(res, '+k', label='values')
        plt.legend()
        plt.show()

    def graph_color(self, save_link: str = ""):
        #  x = np.array(sorted(self.data.raw_data, key=self.data.func))[:, 0]
        #  y = np.array(sorted(self.data.raw_data, key=self.data.func))[:, 1]
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

    def graph3d(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        x = self.data.question.data[:, 0]
        y = self.data.question.data[:, 1]
        z_exp = self.data.expected.data
        z_val = [self(val) for val in self.data.question.data]

        ax.plot(x, y, z_exp, 'or', label='expected')
        ax.plot(x, y, z_val, '+k', label='values')
        plt.legend()
        plt.show()
