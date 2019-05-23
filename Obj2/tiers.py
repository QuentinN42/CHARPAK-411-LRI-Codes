"""
Test 1/3 2/3 with random and not linear dataset

@date: 23/05/2019
@author: Quentin Lieumont
"""
from useful.data import Data
from useful.simpleNetwork import SimpleNetwork
import numpy as np


if __name__ == "__main__":
    from random import random as rand
    import matplotlib.pyplot as plt

    n = 100
    get = 0.1

    tab = np.array([i for i in range(n+1) if i < get*n or i > (1-get)*n]) / n
    to_transpose = [
        np.repeat(np.tile(tab, len(tab) ** i), len(tab) ** (2 - i - 1))
        for i in range(0, 2)
    ]
    d = np.transpose(to_transpose)
    print(len(d))

    R2 = []
    error = [i/10 for i in range(11)]
    # print(error)

    for e in error:
        print(e)

        def f(t: iter):
            return t[0]/3 + 2*t[1]/3 + rand()*e

        data = Data(d, f)
        net = SimpleNetwork(func=f, quiet=True)
        R2.append(((net.weights[0] - 1/3) + (net.weights[1] - 2/3))**2)
    # print(R2)
    plt.plot(error, R2, '+k')
    plt.show()
