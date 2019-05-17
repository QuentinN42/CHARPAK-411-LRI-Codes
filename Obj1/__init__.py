"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from useful.choquet import Choquet, ChoquetData, ChoquetNetwork
from useful.functions import plot_color, average, write_json, nmap, std_err
from math import sqrt


def loss_abs(self: ChoquetNetwork, exp: float, ret: float) -> float:
    # abs(1 - sum(self.weights))
    # abs(1 - self.weights @ self.weights)
    return abs(exp - ret)


def score(real: iter, get: iter) -> float:
    return sum([(get[i]-real[i])**2 for i in range(len(real))])


def test_loss(ch: Choquet, loss_f: callable, sort: bool = False, index: int = 0):
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


def test_1(ch: Choquet, real_W: iter, n: int, sort: bool = False):
    # X = np.arange(100)/100
    # res = [[ch(np.array([x, y])) for x in X] for y in X]
    # plot_color(res, X, plot_title="Expected").show()
    ret = []
    for i in range(n):
        print(i, "/", n)
        ret.append(test_loss(ch, loss_abs, sort=sort, index=1))

    """
    from matplotlib import pyplot as plt
    plt.plot(real_W, 'or', label='Expected')
    for i in range(n):
        plt.plot(ret[i], '+k')
    plt.xticks(range(4), labels=["$w_{}$".format(j) for j in range(1, 5)])
    plt.legend()
    plt.title(title)
    plt.ylim(0, 1)
    plt.savefig("data/tmp.png")
    plt.show()
    """
    return ret


def main():
    n = 100
    real_W = np.array([0.25, 0.25, 0.25, 0.25])
    v1 = real_W[:2]
    v2 = np.array([real_W[2]])
    v3 = np.array([real_W[3]])
    ch = Choquet(v1, v2, v3)
    WF = nmap(np.array, test_1(ch, real_W, n, False))
    WT = nmap(np.array, test_1(ch, real_W, n, True))

    from matplotlib import pyplot as plt
    plt.plot(real_W, 'or', label='Expected')

    moy_WF = list(map(average, np.transpose(WF)))
    moy_WT = list(map(average, np.transpose(WT)))

    err_WF = list(map(std_err, WF))
    err_WT = list(map(std_err, WT))

    err_WFb = [moy_WF[i] - err_WF[i] for i in range(4)]
    err_WFh = [moy_WF[i] + err_WF[i] for i in range(4)]
    err_WTb = [moy_WT[i] - err_WT[i] for i in range(4)]
    err_WTh = [moy_WT[i] + err_WT[i] for i in range(4)]

    plt.plot(err_WFh, 'vk')
    plt.plot(moy_WF, '+k', label="Random")
    plt.plot(err_WFb, '^k')

    plt.plot(err_WTh, 'vb')
    plt.plot(moy_WT, '+b', label="Triées")
    plt.plot(err_WTb, '^b')

    plt.xticks(range(4), labels=["$w_{}$".format(j) for j in range(1, 5)])
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig("data/tmp.png")
    plt.show()
    # write_json("data/json/abs.json", scores)


if __name__ == '__main__':
    main()
