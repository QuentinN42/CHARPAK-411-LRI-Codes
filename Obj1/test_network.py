# -*- coding: utf-8 -*-
"""
Generation des tests

@date: 04/05/2019
@author: Quentin Lieumont
"""
from useful.choquet import Choquet, ChoquetData, ChoquetNetwork


def loss_abs(self: ChoquetNetwork, exp: float, ret: float) -> float:
    # abs(1 - sum(self.weights))
    return abs(exp - ret)


def score(real: iter, get: iter) -> float:
    return sum([(get[i]-real[i])**2 for i in range(len(real))])


def test_1(ch: Choquet, loss_f: callable, sort: bool = False, size: int = 10000):
    """
    test a network
    :param ch: func
    :param loss_f: the loss
    :param sort: sort data set
    :param size: data set size
    :return: weights
    """
    chd = ChoquetData(func=ch, n=size, sort=sort)
    net = ChoquetNetwork(chd, quiet=True, split_ratio=0.5, loss_func=loss_f)
    wts = list(map(lambda w: w/sum(net.weights), net.weights))
    return wts


def test_n(ch: Choquet, n: int, loss_f: callable, sort: bool = False, quiet: bool = False, size: int = 10000):
    """
    test one network n times with the loss func
    :param quiet: quiet mod: no output
    :param loss_f: loss function
    :param ch: func
    :param n: number of iterations
    :param sort: sort data set
    :param size: data set size
    :return: weights list
    """
    ret = []
    for i in range(n):
        if quiet:
            print(i, "/", n)
        ret.append(test_1(ch, loss_f, sort=sort, size=size))
    return ret
