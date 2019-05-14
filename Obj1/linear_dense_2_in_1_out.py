# -*- coding: utf-8 -*-
from keras import layers
from useful import Network, Data, generate


def run(d: Data) -> Network:
    network = Network(d)
    network.model.add(layers.Dense(1, activation='linear', input_dim=2, use_bias=False))
    network.build()

    network.train(1)

    return network


if __name__ == '__main__':
    # training for a + b
    run(Data(generate(), sum))
