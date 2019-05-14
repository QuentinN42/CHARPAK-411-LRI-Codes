# -*- coding: utf-8 -*-
from keras import layers
from useful import Network, Data


def run(d: Data) -> Network:
    network = Network(d)
    network.model.add(layers.Dense(1, activation='linear', input_dim=3, use_bias=False))
    network.build()

    network.train(0.003, False)

    return network
