"""
Define the choquet function.
Define a network that can regress a choquet function.

@date: 15/05/2019
@author: Quentin Lieumont
"""
from useful.simpleNetwork import SimpleNetwork
import numpy as np
from useful.functions import nmap, two_by_two, generate, same_len
from useful.data import Data


class Choquet:
    def __init__(self,
                 w: np.array,
                 w_min: np.array,
                 w_max: np.array):
        if not same_len([w, w_min, w_max]):
            raise AttributeError('Vectors must have the same length')
        else:
            self.dim = len(w)
            self.w = w
            self.w_m = w_min
            self.w_M = w_max

    def __call__(self, x: np.array) -> float:
        x_m = nmap(min, two_by_two(x))
        x_M = nmap(max, two_by_two(x))
        return self.w @ x + self.w_M @ x_M + self.w_m @ x_m


class ChoquetData(Data):
    def __init__(self, tab: np.array = None, func: Choquet = None, expected: np.array = None):
        if not tab:
            if func:
                tab = generate(dim=func.dim)
            else:
                raise AttributeError("Can't extract dimension")
        if func:
            super().__init__(tab, func)
        elif expected:
            def _func(inp):
                if inp in expected:
                    return expected[inp]
                else:
                    return 0
            super().__init__(tab, _func)
        else:
            raise AttributeError('func or awnsers needed')


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
        super().__init__(data, data.func, data.n_dim, use_bias, activation, split_ratio, validate)


def demo():
    v1 = np.array([0.5, 5, 10])
    v2 = np.array([1, 5, 2])
    v3 = np.array([0.2, 0.1, 0.1])
    ch = Choquet(v1, v2, v3)
    chd = ChoquetData(func=ch)
    net = ChoquetNetwork(chd)


if __name__ == '__main__':
    demo()
