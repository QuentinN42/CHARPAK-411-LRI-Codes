from Obj1.test_network import run as simple_network
from useful import generate, Network, Data, moy


def demo() -> Network:
    net = simple_network(Data(generate(dim=3), moy))
    return net
