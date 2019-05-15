"""
Data class

@date: 15/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from useful.functions import shuffle, nmap


class _LearningData:
    def __init__(self, tab: np.array):
        self.data = tab
        self.training = np.array([])
        self.testing = np.array([])

    def __len__(self) -> int:
        return len(self.data)

    def split(self, index: int) -> None:
        self.training, self.testing = self.data[:index], self.data[index:]


class Data:
    def __init__(self, tab: np.array, func: callable):
        self.func = func
        if len(tab) == 0:
            raise EnvironmentError('Data must have some data')
        self.raw_data = tab
        self.question = _LearningData(shuffle(self.raw_data))
        self.expected = _LearningData(nmap(self.func, self.question.data))

    def __len__(self) -> int:
        return len(self.question)

    def split(self, learning_set_ratio: float = 0.5) -> None:
        split_at = int(learning_set_ratio * len(self))
        self.question.split(split_at)
        self.expected.split(split_at)

    @property
    def n_dim(self):
        return len(self.raw_data[0])
