"""
Declaration des graphiques

@date: 04/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from useful.functions import average, nmap, std_err
from useful.choquet import Choquet
from .test_network import test_n, loss_abs
from matplotlib import pyplot as plt


def sort_unsort() -> None:
    n = 100
    real_W = np.array([0.25, 0.25, 0.25, 0.25])
    v1 = real_W[:2]
    v2 = np.array([real_W[2]])
    v3 = np.array([real_W[3]])
    ch = Choquet(v1, v2, v3)
    WF = nmap(np.array, test_n(ch, n, loss_abs, False))
    WT = nmap(np.array, test_n(ch, n, loss_abs, True))

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


def test(size_from: int,
         size_to: int,
         size_ech: int,

         loss_test_list: list,
         number_of_learning: int
         ) -> None:
    """
    #TODO: 4 tests : (contrainte [0,1]/sans)/(random/trié)
    #TODO: variation de l'ectart type en fonction de la taille du training 1000 -> 50000
    """
    # building data :
    labels = ["Random", "Trié"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    sizes = range(size_from, size_to, size_ech)
    real_W = np.array([0.25, 0.25, 0.25, 0.25])
    v1 = real_W[:2]
    v2 = np.array([real_W[2]])
    v3 = np.array([real_W[3]])
    ch = Choquet(v1, v2, v3)
    fig, ax = plt.subplots()
    ax.set_ylim(0)

    # looping learn
    for size in sizes:
        for loss in loss_test_list:
            for b in [True, False]:
                lab = "Loss : {f}, {trie}".format(f=loss.__name__, trie=labels[int(b)])
                W = nmap(np.array, test_n(ch, number_of_learning, loss, b))

                err_W = nmap(std_err, W)
                err_ave = average(err_W)
                err_err = std_err(err_W)
                err_avb = err_ave - err_err
                err_avh = err_ave + err_err

                c = colors[loss_test_list.index(loss)*2 + int(b)]

                plt.plot(err_avh, 'v' + c)
                plt.plot(err_ave, '+' + c, label=lab)
                plt.plot(err_avb, '^' + c)
    fig.show()
