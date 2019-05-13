# -*- coding: utf-8 -*-
from keras import layers
import numpy as np
from useful import Network, Data


def generate() -> np.array:
    return np.array([[i/100, j/100] for i in range(10) for j in range(10)])


d = Data(generate(), sum)

network = Network(d)
network.model.add(layers.Dense(1, activation='linear', input_dim=2, use_bias=False))
network.build()

network.train()
network.graph()
