"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from useful.choquet import Choquet, ChoquetData, ChoquetNetwork


real_W = np.array([0.5, 0.25, 0.1, 0.15])


def loss(self: ChoquetNetwork, exp: float, ret: float) -> float:
    return abs(exp - ret) + sum(self.weights)


def test_loss(loss_f: callable):
    v1 = real_W[:2]
    v2 = np.array([real_W[2]])
    v3 = np.array([real_W[3]])
    ch = Choquet(v1, v2, v3)
    chd = ChoquetData(func=ch, n=100000)
    net = ChoquetNetwork(chd, split_ratio=0.8, loss_func=loss_f)
    return list(map(lambda w: w/sum(net.weights), net.weights))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.plot(real_W, 'or')
    for i in range(4):
        print(i, "="*50)
        ret = test_loss(loss)
        plt.plot(ret, '+', label="abs")
    plt.xticks(range(4), labels=["$w_{}$".format(j) for j in range(1, 5)])
    plt.legend()
    plt.savefig("data/tmp.png")
    plt.show()

