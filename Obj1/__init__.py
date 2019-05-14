"""
Creation d'un réseau de neurones pour regresser une integrale de choquet

@date: 04/05/2019
@author: Quentin Lieumont
"""
from Obj1.test_network import run as simple_network
from useful import generate, Network, Data, moy
from keras.utils import plot_model


def demo() -> Network:
    net = simple_network(Data(generate(10, 10, dim=3), moy))
    plot_model(net.model)
    return net
