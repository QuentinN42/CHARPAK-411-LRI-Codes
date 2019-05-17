"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from useful.choquet import Choquet, ChoquetData, ChoquetNetwork
from useful.functions import plot_color, average, write_json


real_W = np.array([0.5, 0, 0, 0.5])
v1 = real_W[:2]
v2 = np.array([real_W[2]])
v3 = np.array([real_W[3]])
ch = Choquet(v1, v2, v3)


def loss_abs(self: ChoquetNetwork, exp: float, ret: float) -> float:
    # abs(1 - sum(self.weights))
    # abs(1 - self.weights @ self.weights)
    return abs(exp - ret)


def score(real: iter, get: iter) -> float:
    return sum([abs(get[i]-real[i]) for i in range(len(real))])


def test_loss(loss_f: callable, sort: bool = False, index: int = 0):
    global ch
    chd = ChoquetData(func=ch, n=10000, sort=sort)
    net = ChoquetNetwork(chd, quiet=True, split_ratio=0.5, loss_func=loss_f)
    wts = list(map(lambda w: w/sum(net.weights), net.weights))
    # v1 = np.array(wts[:2])
    # v2 = np.array([wts[2]])
    # v3 = np.array([wts[3]])
    # chf = Choquet(v1, v2, v3)
    # X = np.arange(100)/100
    # res = [[chf(np.array([x, y])) for x in X] for y in X]
    # plot_color(res, X, plot_title=str(i)).show()
    return wts


def main():
    n = 5
    # X = np.arange(100)/100
    # res = [[ch(np.array([x, y])) for x in X] for y in X]
    # plot_color(res, X, plot_title="Expected").show()
    ret = []
    for i in range(n):
        print(i, "/", n)
        ret.append(test_loss(loss_abs, sort=False, index=1))

    from matplotlib import pyplot as plt
    plt.plot(real_W, 'or')
    scores = []
    for i in range(n):
        scores.append(score(real_W, ret[i]))
        plt.plot(ret[i], '+', label=scores[i])
    plt.xticks(range(4), labels=["$w_{}$".format(j) for j in range(1, 5)])
    plt.legend()
    plt.savefig("data/tmp.png")
    plt.show()
    write_json("data/json/abs.json", scores)


if __name__ == '__main__':
    main()
