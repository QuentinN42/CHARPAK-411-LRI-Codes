"""
Usefull thinks...

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from keras.models import Sequential
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(e):
    return int(e*100)/100


def shuffle(tab: np.array) -> np.array:
    indices = np.arange(len(tab))
    np.random.shuffle(indices)
    return tab[indices]


def generate(n: int = 10, div: int = 100) -> np.array:
    return np.array([[i/div, j/div] for i in range(n) for j in range(n)])


def moy(tab: np.array) -> float:
    return sum(tab)/len(tab)


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
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, split_ratio: float = 0.5):
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

    def normalize(self, val: float) -> float:
        r = self.predictions
        return (val+abs(min(r)))*max(self.data.expected.data)/(max(r) - min(r))

    @property
    def normalized_predictions(self):
        return [self.normalize(e) for e in self.predictions]

    def __call__(self, inp):
        return self.normalize(self.predict(inp))

    def graph(self):
        exp = sorted(self.data.expected.data)
        questions = sorted(self.data.question.data, key=self.data.func)
        res = [self.predict(q) for q in questions]
        plt.plot(exp, '-r', label='expected')
        plt.plot(res, '+k', label='values')
        plt.legend()
        plt.show()

    def graph_color(self, debug: bool = False):
        #  x = np.array(sorted(self.data.raw_data, key=self.data.func))[:, 0]
        #  y = np.array(sorted(self.data.raw_data, key=self.data.func))[:, 1]
        x = np.arange(20)/200
        y = np.arange(20)/200
        z_exp = [[self.data.func([_y, _x]) for _y in y] for _x in x]

        if debug:
            for i in range(len(x)):
                for j in range(len(y)):
                    print('{} + {} = {}'.format(x[i], y[j], z_exp[i][j]))

        fig, ax = plt.subplots()
        im = ax.imshow(z_exp)
        fig.tight_layout()
        plt.colorbar(im)
        plt.show()

        z_val = [[self.predict([_y, _x]) for _y in y] for _x in x]

        fig, ax = plt.subplots()
        im = ax.imshow(z_val)
        fig.tight_layout()
        plt.colorbar(im)
        plt.show()

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
