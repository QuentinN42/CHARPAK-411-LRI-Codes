"""
Define the choquet function.
Define a network that can regress a choquet function.

@date: 15/05/2019
@author: Quentin Lieumont
"""
import math
from useful.simpleNetwork import SimpleNetwork
import numpy as np
from useful.functions import nmap, two_by_two
from useful.data import Data


# TODO : check the w_min/w_max len (Issue#1)
# TODO : set self.n_dim at the right value
class Choquet:
    def __init__(self,
                 w: np.array,
                 w_min: np.array,
                 w_max: np.array):
        if len(w_min) == len(w_max) == (len(w)*(len(w)-1))/2:
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


def choquet_generate(ch: Choquet, n: int = 100, debug: bool = False) -> dict:
    """
    Generate n vectors input and expected by the ch function
    :param ch: Choquet function
    :param n: number of vectors
    :param debug: See function work progress
    :return: dict : {"expected": [...], "question": [...]}
    """
    que = []
    exp = []
    print_index = nmap(int, np.arange(100)*n/100)
    for i in range(n):
        if debug and i in print_index:
            print("Building : {}%...".format(str(int(i*100/n)).zfill(2)))
        random_vect = np.random.rand(int(math.sqrt(ch.n_dim)))
        que.append(random_vect.tolist())
        exp.append(ch(random_vect))
    return {"question": que, "expected": exp}


class ChoquetData(Data):
    func: Choquet

    def __init__(self, func: Choquet, n: int = None, debug: bool = False):
        if n is not None:
            dico = choquet_generate(func, n, debug)
        else:
            dico = choquet_generate(func, debug=debug)
        que = nmap(np.array, dico['question'])
        exp = nmap(np.array, dico['expected'])
        self.func = func
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

                 # Layer options
                 use_bias: bool = False,
                 activation: str = 'linear',

                 # Training options
                 split_ratio: float = 0.5,
                 validate: bool = True
                 ):
        super().__init__(data, n_dim=data.n_dim, use_bias=use_bias,
                         activation=activation, split_ratio=split_ratio, validate=validate)


def demo():
    v1 = np.array([0.3, 0.4])
    v2 = np.array([0.1])
    v3 = np.array([0.2])
    ch = Choquet(v1, v2, v3)
    chd = ChoquetData(func=ch, debug=True)
    net = ChoquetNetwork(chd)
    return net
