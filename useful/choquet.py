"""
Define the choquet function.
Define a network that can regress a choquet function.

@date: 15/05/2019
@author: Quentin Lieumont
"""
import math
from useful.simpleNetwork import SimpleNetwork
import numpy as np
from useful.functions import nmap, two_by_two, readable
from useful.data import Data


class Choquet:
    def __init__(self,
                 w: np.array,
                 w_min: np.array,
                 w_max: np.array):
        if len(w_min) == len(w_max) == (len(w) * (len(w) - 1)) / 2:
            self.n_dim = len(w) ** 2
            self.w = w
            self.w_m = w_min
            self.w_M = w_max
        else:
            raise AttributeError('Length not match')

    @property
    def W(self):
        return np.concatenate((self.w, self.w_m, self.w_M))

    @staticmethod
    def pre_call(x) -> np.ndarray:
        return np.concatenate((x, two_by_two(x, min), two_by_two(x, max)))

    def __call__(self, x: np.array) -> float:
        return Choquet.pre_call(x) @ self.W


def choquet_generate(ch: Choquet, n: int = 100, debug: bool = False, sort: bool = False) -> dict:
    """
    Generate n vectors input and expected by the ch function
    :param ch: Choquet function
    :param n: number of vectors
    :param debug: See function work progress
    :param sort: sort vectors
    :return: dict : {"expected": [...], "question": [...]}
    """
    que = []
    exp = []
    print_index = nmap(int, np.arange(100) * n / 100)
    for i in range(n):
        if debug and i in print_index:
            print("Building : {}%...".format(str(int(i * 100 / n)).zfill(2)))
        random_vect = np.random.rand(int(math.sqrt(ch.n_dim)))
        que.append(random_vect.tolist())
        if not sort:
            exp.append(ch(random_vect))
    if sort:
        if debug:
            print("Sorting data")
        que.sort(key=lambda t: t[0])
        if debug:
            print("Mapping func on questions")
        exp = list(map(ch, que))
    return {"question": que, "expected": exp}


class ChoquetData(Data):
    func: Choquet

    def __init__(self, func: Choquet, n: int = None, debug: bool = False, sort: bool = False):
        if n is not None:
            dico = choquet_generate(func, n, debug, sort)
        else:
            dico = choquet_generate(func, debug=debug, sort=sort)
        que = nmap(np.array, dico['question'])
        exp = nmap(np.array, dico['expected'])
        self.func: Choquet = func
        super().__init__(tab=que, expected=exp)

    @property
    def n_dim(self):
        if self.func is not None:
            return self.func.n_dim

    @property
    def question_data(self):
        return nmap(Choquet.pre_call, self.question.data)

    @property
    def question_training(self):
        return nmap(Choquet.pre_call, self.question.training)

    @property
    def question_testing(self):
        return nmap(Choquet.pre_call, self.question.testing)

    @property
    def expected_data(self):
        return self.expected.data

    @property
    def expected_training(self):
        return self.expected.training

    @property
    def expected_testing(self):
        return self.expected.testing


class ChoquetNetwork(SimpleNetwork):
    def __init__(self,
                 # Data initialisation
                 data: ChoquetData,
                 quiet: bool = False,

                 # Layer options
                 use_bias: bool = False,
                 activation: str = 'linear',

                 # Training options
                 split_ratio: float = 0.5,
                 loss_func: callable = None,
                 validate: bool = True
                 ):
        if loss_func:
            def _loss_func(e, r):
                return loss_func(self, e, r)
            super().__init__(data, quiet=quiet, n_dim=data.n_dim,
                             use_bias=use_bias, activation=activation, allow_neg=False,
                             split_ratio=split_ratio, loss_func=_loss_func, validate=validate)
        else:
            super().__init__(data, quiet=quiet, n_dim=data.n_dim,
                             use_bias=use_bias, activation=activation, allow_neg=False,
                             split_ratio=split_ratio, validate=validate)

    def predict(self, inp):
        return super().predict(self.data.func.pre_call(inp))
