"""
Usefull functions

@date: 15/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import json


def get_json(link: str) -> dict:
    return json.load(open(link, "r"))


def write_json(link: str, data) -> None:
    _json = json.dumps(data, sort_keys=True, separators=(",", ": "))
    with open(link, "w") as f:
        for l in _json.split("\n"):
            f.write(l + "\n")


def title(t: str):
    print("\n" * 5)
    print("=" * 20 + t + "=" * 20)


def white_space(func):
    def inner(*args, **kwargs):
        print("\n" * 5)
        func(*args, **kwargs)
        print("\n" * 5)

    return inner


def nmap(func: callable, l) -> np.array:
    """
    Map build-in function for numpy
    :param func: the function to map like f
    :param l: a list like [a, b, c] (or numpy array)
    :return: numpy.array(f(a), f(b), f(c))
    """
    return np.array(list(map(func, l)))


def two_by_two(vector: iter, func: callable) -> np.array:
    out = np.array([])
    length = len(vector)
    for i in range(length):
        for j in range(i + 1, length):
            out = np.append(out, func(vector[[i, j]]))
    return out


def same_len(vectors: iter) -> bool:
    """
    have all vectors the same length ?
    :param vectors: a list of Sized
    :return: if all vectors have the same length
    """
    lengths = list(map(len, vectors))
    return min(lengths) == max(lengths)


def readable(e) -> str:
    """
    format 0.15424548 in to 0.15
    :return: readable number
    """
    return str(int(e * 100) / 100)


def is_negative(n) -> int:
    if n < 0:
        return 1
    else:
        return 0


def shuffle(tab: iter) -> np.array:
    """
    shuffle an array
    :param tab: an array
    :return: the array with the same element but in different order
    """
    indices = np.arange(len(tab))
    np.random.shuffle(indices)
    return tab[indices]


def generate(n: int = 100, div: int = 100, dim: int = 2) -> np.array:
    """
    Generate the list of all vectors at [dim] dimensions from [-n/div] to [n/div] with a step of [1/div]
    """
    t = np.arange(n) / div
    tab = np.concatenate((t[:-1] - max(t), t))
    to_transpose = [
        np.repeat(np.tile(tab, len(tab) ** i), len(tab) ** (dim - i - 1))
        for i in range(0, dim)
    ]
    return np.transpose(to_transpose)


def idem_items(t: iter) -> bool:
    """
    :return: if all items are sames
    """
    return max(t) == min(t)


def average(tab: iter) -> float:
    """
    return the average of a table
    """
    return sum(tab) / len(tab)


def std_err(tab: iter):
    e = 0
    m = average(tab)
    for i in tab:
        e += (i - m) ** 2
    return sqrt(e / len(tab))


def plot_color(
    z: iter, x: iter = None, y: iter = None, nb_ticks: int = 5, plot_title: str = ""
) -> plt:
    """
    plot a 2D colored graph of a 2D array
    :param z: the 2D array
    :param x: x axis label
    :param y: y axis label
    :param nb_ticks: number of labels over 1 axis
    :param plot_title: the plot title
    :return: the pyplot (if you want to save it)
    """
    fig, ax = plt.subplots()
    im = ax.imshow(z)
    ax.plot([len(z) - 0.5, -0.5], [-0.5, len(z) - 0.5], "-k")
    if plot_title is not "":
        ax.title._text = plot_title
    fig.tight_layout()
    fig.colorbar(im)
    if x is not None:
        if y is None:
            y = x
        ticks_pos_x = nmap(int, np.arange(nb_ticks) * (len(x) - 1) / (nb_ticks - 1))
        ticks_pos_y = nmap(int, np.arange(nb_ticks) * (len(y) - 1) / (nb_ticks - 1))
        ax.set_xticks(ticks_pos_x)
        ax.set_yticks(ticks_pos_y)
        ax.set_xticklabels([x[i] for i in ticks_pos_x])
        ax.set_yticklabels([y[i] for i in ticks_pos_y])
    return fig


def plot_3d(x, y, expected, result) -> plt:
    """
    plot expected and result value over 2 axis
    :param x: x array
    :param y: y array
    :param expected: 2D array of expected values
    :param result: 2D array of returned values
    :return: the pyplot
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, expected, "or", label="expected")
    ax.plot(x, y, result, "+k", label="values")
    plt.legend()
    return plt


def history_plot(history: dict, key: str, validation_set: bool) -> plt:
    """
    plot the learning history
    :param history: data
    :param key: 'loss' or 'acc'
    :param validation_set: if the validation set has been used
    :return:
    """
    plt.plot(history[key], label="Train")
    if validation_set:
        plt.plot(history["val_" + key], label="Test")
    plt.title("Model " + key)
    plt.ylabel(key.capitalize())
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    return plt
