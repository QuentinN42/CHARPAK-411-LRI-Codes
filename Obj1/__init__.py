from Obj1.linear_dense_2_in_1_out import run as simple_network
from useful import generate, Network, Data, moy


def demo() -> Network:
    net = simple_network(Data(generate(), sum))
    net.graph_color()
    return net
