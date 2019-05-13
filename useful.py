"""
Usefull thinks...

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from keras.models import Sequential
from matplotlib import pyplot as plt


def f(e):
    return int(e*100)/100


def shuffle(tab: np.array) -> np.array:
    indices = np.arange(len(tab))
    np.random.shuffle(indices)
    return tab[indices]


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
        if len(tab) == 0:
            raise EnvironmentError('Data must have some data')
        self.question = LearningData(shuffle(tab))
        self.expected = LearningData(np.array([func(e) for e in self.question.data]))

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
    def normalized_prediction(self):
        return [self.normalize(e) for e in self.predictions]

    def __call__(self, inp):
        return self.normalize(self.predict(inp))

    def graph(self):
        plt.plot(sorted(self.normalized_prediction), '+k', label='values')
        plt.plot(sorted(self.data.expected.data), '-r', label='expected')
        plt.legend()
        plt.show()
