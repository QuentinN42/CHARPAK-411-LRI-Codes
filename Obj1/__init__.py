"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

#TODO: bruit ?
#TODO: Kangle

@date: 04/05/2019
@author: Quentin Lieumont
"""
from Obj1.plots import loss_test as test
from Obj1.test_network import loss_abs, loss_abs_norm


def main():
    # TODO: 4 tests : (contrainte [0,1]/sans)
    # TODO: variation de l'ectart type en fonction de la taille du training 1000 -> 10 000
    test([10, 50, 100, 500, 1000, 5000, 10000], [loss_abs, loss_abs_norm], number_of_learning=20)


if __name__ == '__main__':
    main()
