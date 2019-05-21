"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

#TODO: bruit ?
#TODO: Kangle

@date: 04/05/2019
@author: Quentin Lieumont
"""
from Obj1.plots import loss_learn_test as test
from Obj1.test_network import loss_abs, loss_abs_norm
from useful.functions import nmap
import numpy as np


def main():
    dix = 10 ** np.arange(1, 3)
    n = sorted(nmap(int, np.concatenate((dix / 4, 2 * dix / 4, 3 * dix / 4, dix))))
    print(n)
    test(number_of_learning=n, loss_test_list=[loss_abs, loss_abs_norm], size=10)


if __name__ == "__main__":
    main()
