"""
Real dataset from https://www.kaggle.com/harlfoxem/housesalesprediction

@date: 23/05/2019
@author: Quentin Lieumont
"""
import math
import numpy as np
from matplotlib import pyplot as plt
from useful.functions import get_json, write_json, history_plot, my_zip, std_err, average
from useful.simpleNetwork import SimpleNetwork
from useful.data import Data
import os
from scipy.optimize import curve_fit


#%%  Class Houses


class Houses:
    def __init__(self, link, debug: bool = False):
        self.header_from_int = {i: e for i, e in enumerate(get_json(link + '/header.json')[2:])}
        self.header_from_str = {v: k for k, v in self.header_from_int.items()}
        if debug:
            for k, v in self.header_from_str.items():
                print(k, v)
        self.raw_data: np.ndarray = np.genfromtxt(link + '/raw.csv', delimiter=',')[:, 2:]
        for i in range(len(self.raw_data[:, self.header_from_str["yr_renovated"]])):
            if self.raw_data[i, self.header_from_str["yr_renovated"]] == 0:
                self.raw_data[i, self.header_from_str["yr_renovated"]] = \
                    self.raw_data[i, self.header_from_str["yr_built"]]

        # self.raw_data = self.raw_data[:, :-5]
        # self.raw_data = np.delete(self.raw_data, (7, 12), 1)
        self.norm_data: np.ndarray = np.transpose(np.array(
            [
                (np.array(t)-min(t))/(max(t)-min(t))
                for t in np.transpose(self.raw_data).tolist()
            ]
        ))

    def __getitem__(self, item):
        if type(item) is str:
            return self.raw_data[:, self.header_from_str[item]]
        elif type(item) is int:
            return self.raw_data[:, item]
        else:
            raise AttributeError(f"Unrecognised data type : {type(item)}")

    def __call__(self, item):
        if type(item) is str:
            return self.norm_data[:, self.header_from_str[item]]
        elif type(item) is int:
            return self.norm_data[:, item]
        else:
            raise AttributeError(f"Unrecognised data type : {type(item)}")

    def color_scatter(self, c, xy=None, norm: bool = False) -> plt.figure:
        fig, ax = plt.subplots()
        if xy:
            cm = ax.scatter(xy[0], xy[1], s=8, c=self[c] if not norm else self(c))
        else:
            cm = ax.scatter(self.lat, self.long, s=8, c=self[c] if not norm else self(c))
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Longitude")
        fig.colorbar(cm)
        fig.suptitle(
            f"{c if type(c) is not int else self.header_from_int[c]} \
            {'Real values' if norm else 'Normalized values'}",
        )
        fig.tight_layout()

        return fig

    def plot_all_f_price(self):
        for i in range(len(self.raw_data[0])):
            fig, ax = plt.subplots()
            ax.plot(self[i], self[0], '+')
            ax.set_ylabel("Price")
            ax.set_xlabel(self.header_from_int[i])
            fig.show()

    # ======================== # Properties # ======================== #

    @property
    def lat(self):
        return self["lat"]

    @property
    def long(self):
        return self["long"]


#%%  Class HouseLearner


class HouseLearner(SimpleNetwork):
    def __init__(self, houses: Houses, epochs: int = 10, q: bool = True):
        self.houses = houses
        d: Data = Data(tab=houses.norm_data[:, 1:], expected=houses("price"))
        super().__init__(
            data=d,
            n_dim=len(houses.norm_data[0, 1:]),
            epochs=epochs,
            quiet=q
        )


#%%  Main functions


def build(n_epochs: int = 10, n_build: int = 1, save_folder: str = "data/Obj2/real/default"):
    """
    Build and save learning
    :param n_epochs:
    :param n_build:
    :param save_folder: folder to save the file
    :return:
    """
    if os.path.isdir(save_folder):
        print(f"Ready to save {n_build} files in {save_folder}")
    else:
        raise FileExistsError(f"Path {save_folder} not found or not a directory !")
    for n in range(n_build):
        print(f"Bluid number {n+1} over {n_build} :")
        h = Houses("learning_data/kc_house")
        net = HouseLearner(h, epochs=n_epochs)
        _id = int(len(os.listdir(save_folder))/3)
        write_json(f"{save_folder}/val_loss{_id}.json", net.history['val_loss'])
        write_json(f"{save_folder}/loss{_id}.json", net.history['loss'])
        write_json(f"{save_folder}/weights{_id}.json", net.weights.tolist())


def history_from_file(n: int = -1, folder: str = "data/Obj2/real/default") -> plt.Figure:
    """
    Plot form data/Obj2/real/...n.json
    :param folder: folder with lossXX.json file
    :param n: data index if -1 the last index is returned
    :return:
    """
    if type(n) is int:
        n: int
        _id = list(range(int(len(os.listdir(folder)) / 3)))[n]
        dico = {
            "loss": get_json(f"{folder}/loss{_id}.json"),
            "val_loss": get_json(f"{folder}/val_loss{_id}.json")
        }
        return history_plot(dico, "loss", True)

    elif type(n) is slice:
        _ids = list(range(int(len(os.listdir(folder)) / 3)))[n]
        dicos = [
            {
                "loss": get_json(f"{folder}/loss{_id}.json"),
                "val_loss": get_json(f"{folder}/val_loss{_id}.json")
            }
            for _id in _ids
        ]
        return history_plot(dicos, "loss", True)
    else:
        raise AttributeError(f'n type error: {type(n)} with value {n}')


def weights_from_file(n: slice = -1, labels_link: str = None, folder: str = "data/Obj2/real/default") -> plt.Figure:
    """
    Plot form data/Obj2/real/...n.json
    :param folder: folder with weightsXX.json file
    :param labels_link: link to show labels
    :param n: data index if -1 the last index is returned
    :return:
    """
    fig, ax = plt.subplots()
    if type(n) is int:
        n: int
        _id = list(range(int(len(os.listdir(folder)) / 3)))[n]
        weights = get_json(f"{folder}/weights{_id}.json")
        list_n = list(range(len(weights)))
        norm_weights = [e / len(weights) for e in weights]
        norm_weights_a = [abs(e) for e in norm_weights]
        ax.bar(
            list_n,
            norm_weights_a,
            color=['red' if e < 0 else 'blue' for e in norm_weights]
        )

    elif type(n) is slice:
        _ids = list(range(int(len(os.listdir(folder)) / 3)))[n]
        weightss = my_zip([get_json(f"{folder}/weights{_id}.json") for _id in _ids])
        list_n = list(range(len(weightss)))
        avs = list(map(average, weightss))
        errs = list(map(std_err, weightss))
        ax.bar(
            list_n,
            [abs(e) for e in avs],
            color=['red' if e < 0 else 'blue' for e in avs],
            yerr=errs
        )
    else:
        raise AttributeError(f'n type error: {type(n)} with value {n}')

    if labels_link:
        names = get_json(labels_link)[3:]

        ax.set_xticks(list_n)
        ax.set_xticklabels(names, rotation=90)
    return fig


#%%  Regress abstract func


def fit(func, x, y):
    param, pcov = curve_fit(func, x, y, maxfev=50000)

    pred = func(x, *param)

    R2 = 1 - np.var(pred - y)/np.var(y)

    return R2, param


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Regress_func:
    act: bool = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Abstract class')


#%%  Regress func


class deg0(Regress_func):
    def __call__(self, x, a):
        return a


class deg1(Regress_func):
    def __call__(self, x, a, b):
        return a*x+b


class deg2(Regress_func):
    def __call__(self, x, a, b, c):
        return a*x**2+b*x+c


class deg2_0(Regress_func):
    def __call__(self, x, a, b, c):
        return a*x**2+b*x


class deg3(Regress_func):
    def __call__(self, x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d


class inv(Regress_func):
    def __call__(self, x, a, b, c, d):
        return (a*x+b)/(c*x+d)


class e(Regress_func):
    def __call__(self, x, a, b, c, d):
        return a*np.exp(b*x+c)+d


class root(Regress_func):
    def __call__(self, x, a, b):
        return np.sqrt(a*x+b)


class log(Regress_func):
    def __call__(self, x, a, b):
        return a*np.log(b*x)


class sigm(Regress_func):
    def __call__(self, x, a, b):
        return a*sigmoid(b*x)


class gaussian(Regress_func):
    def __call__(self, x, mu, sig):
        return np.exp(-((x - mu)/sig)**2/2)


#%%  Main


if __name__ == "__main__":
    link = "https://www.kaggle.com/harlfoxem/housesalesprediction"
    print(f"Data from {link}.")

    """
    s = slice(6, None)
    history_from_file(
        folder="data/Obj2/real/default",
        n=s
    ).show()
    weights_from_file(
        folder="data/Obj2/real/default",
        n=s,
        labels_link="learning_data/kc_house/header_remove1.json"
    ).show()
    """

    h = Houses("learning_data/kc_house")
    fs = {f.__name__: f() for f in Regress_func.__subclasses__() if f.act is True}

    prices = h(0)
    length = len(prices)
    li = list(range(len(h.raw_data[0])))[1:]

    i = 1
    min_dist = 0.01
    var = h(i)
    data = sorted([np.array(e) for e in np.transpose((var, prices)).tolist()], key=lambda e: np.linalg.norm(e))

    data2 = []
    prec = -1

    while data:
        if 100-int(len(data)/length*100) > prec:
            prec = 100-int(len(data)/length*100)
            print(prec)

        k = 1
        length_k = len(data)
        poped = False

        while k < length_k:
            if data[0] @ data[k] <= min_dist:
                data2.append(data.pop(k))
                data2.append(data.pop(0))
                poped = True
                break
            else:
                k += 1
        if not poped:
            data.pop(0)

    fig, ax = plt.subplots()
    ax.plot(var, prices, '+b')
    ax.plot([e[0] for e in data2], [e[1] for e in data2], '+k')

    ax.set_ylabel("Price")
    ax.set_xlabel(h.header_from_int[i])
    leg: plt.legend = ax.legend()
    fig.show()

    """
    for i in li:
        max_name: str = ""
        max_r2: float = 0.
        max_param = None
        max_f = None
        fig, ax = plt.subplots()
        ax.plot(h[i], prices, '+')

        for name, f in fs.items():
            # print(h[i].size, '/', h[j].size)
            try:
                r2, vals = fit(f, h[i], prices)
            except RuntimeWarning:
                pass
            except RuntimeError as e:
                # print(name, ': Error:', e)
                pass
            else:
                if r2 > 0.3:
                    if r2 > max_r2:
                        max_r2 = r2
                        max_name = name
                        max_param = vals
                        max_f = f
                    # print(f"{h.header_from_int[0]} | {h.header_from_int[i]} | {name} | {r2}")
        if max_r2 != 0.:
            print(f"For {h.header_from_int[i]}: Best function {max_name} with R2={max_r2}")
            x = np.linspace(min(h[i]), max(h[i]))
            ax.plot(x, max_f(x, *max_param), '-', label=f"{max_name}, R2={str(max_r2)[:4]}")
            ax.set_ylabel("Price")
            ax.set_xlabel(h.header_from_int[i])
            leg: plt.legend = ax.legend()
            fig.show()
        else:
            print(f"No function found for {h.header_from_int[i]}")
"""