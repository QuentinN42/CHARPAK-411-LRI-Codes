"""
Define the choquet function.
Define a network that can regress a choquet function.

@date: 15/05/2019
@author: Quentin Lieumont
"""
from useful.simpleNetwork import SimpleNetwork
import numpy as np
from useful.functions import nmap, two_by_two


class Choquet:
    def __init__(self,
                 w: np.array,
                 w_min: np.array,
                 w_max: np.array):
        self.w = w
        self.w_m = w_min
        self.w_M = w_max

    def __call__(self, x: np.array) -> float:
        x_m = nmap(min, two_by_two(x))
        x_M = nmap(max, two_by_two(x))
        return self.w @ x + self.w_M @ x_M + self.w_m @ x_m
