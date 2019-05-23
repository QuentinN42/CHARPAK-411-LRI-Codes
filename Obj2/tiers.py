"""
Test 1/3 2/3 with random and not linear dataset

@date: 23/05/2019
@author: Quentin Lieumont
"""
from useful.data import Data
from useful.simpleNetwork import SimpleNetwork
from useful.functions import average, std_err, generate, get_json
import numpy as np
from random import random as rand

def test_network(f: callable, d: iter) -> float:
    if d is None:
        data = Data(d, f)
    else:
        data = Data(d, f)
    net = SimpleNetwork(data=data, func=f, quiet=True)
    return ((net.weights[0] - 1 / 3) + (net.weights[1] - 2 / 3)) ** 2


def generate_with_hole(n: int = 100, get: float = 0.1):
    tab = np.array([i for i in range(n+1) if i < get*n or i > (1-get)*n]) / n
    to_transpose = [
        np.repeat(np.tile(tab, len(tab) ** i), len(tab) ** (2 - i - 1))
        for i in range(0, 2)
    ]
    return np.transpose(to_transpose)


def build_data(link: str, n: int = 10):
    d = generate()
    error = [i / 10 for i in range(11)]

    for e in error:
        print(e)

        def f(t: iter):
            return t[0] / 3 + 2 * t[1] / 3 + (rand() * 2 - 1) * e

        t = [test_network(f, d) for i in range(n)]
        with open(link, 'a') as file:
            file.write(str([e, t]) + "\n")


if __name__ == "__main__":
    lien = "data/Obj2/tiers/generared.raw"

    # build_data(lien, n=50)

    dico = {}
    with open(lien, 'r') as f:
        for l in f:
            d: dict = eval(l)
            if d[0] in dico.keys():
                dico[d[0]] = dico[d[0]] + d[1]
            else:
                dico[d[0]] = d[1]

    R2 = []
    R2h = []
    R2b = []
    error = dico.keys()

    for t in dico.values():
        a = average(t)
        s = std_err(t)
        R2h.append(a + s)
        R2.append(a)
        R2b.append(a - s)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylim(0)
    ax.set_xlabel("Random range")
    ax.set_ylabel("Std error")

    ax.plot(error, R2h, 'vk')
    ax.plot(error, R2, '+k')
    ax.plot(error, R2b, '^k')
    fig.show()
