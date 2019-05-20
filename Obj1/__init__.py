"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

#TODO: deux fois deux reseaux : (contrainte [0,1]/sans)/(random/trié)
#TODO: variation de l'ectart type en fonction de la taille du training 1000 -> 50000

#TODO: bruit ?
#TODO: Kangle

@date: 04/05/2019
@author: Quentin Lieumont
"""
from .plots import sort_unsort


def main():
    sort_unsort()


if __name__ == '__main__':
    main()
