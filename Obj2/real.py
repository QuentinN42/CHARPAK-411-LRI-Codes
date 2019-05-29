"""
Real dataset from https://www.kaggle.com/harlfoxem/housesalesprediction

@date: 23/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from matplotlib import pyplot as plt
from useful.functions import get_json, write_json, history_plot, my_zip, std_err, average
from useful.simpleNetwork import SimpleNetwork
from useful.data import Data
import os
from scipy.optimize import curve_fit


# %%  Class Houses


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
                (np.array(t) - min(t)) / (max(t) - min(t))
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


# %%  Class HouseLearner


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


# %%  Regress abstract func


def fit(func: callable, x: np.ndarray, y: np.ndarray) -> tuple:
    param, pcov = curve_fit(func, x, y, maxfev=50000)

    pred = func(x, *param)

    R2 = 1 - np.var(pred - y) / np.var(y)

    return R2, param


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Regress_func:
    act: bool = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('Abstract class')


# %%  Regress func


class deg0(Regress_func):
    def __call__(self, x, a):
        return a


class deg1(Regress_func):
    def __call__(self, x, a, b):
        return a * x + b


class deg2(Regress_func):
    def __call__(self, x, a, b, c):
        return a * x ** 2 + b * x + c


class deg2_0(Regress_func):
    def __call__(self, x, a, b, c):
        return a * x ** 2 + b * x


class deg3(Regress_func):
    def __call__(self, x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d


class inv(Regress_func):
    def __call__(self, x, a, b, c, d):
        return (a * x + b) / (c * x + d)


class e(Regress_func):
    def __call__(self, x, a, b, c, d):
        return a * np.exp(b * x + c) + d


class root(Regress_func):
    def __call__(self, x, a, b):
        return np.sqrt(a * x + b)


class log(Regress_func):
    def __call__(self, x, a, b):
        return a * np.log(b * x)


class sigm(Regress_func):
    def __call__(self, x, a, b):
        return a * sigmoid(b * x)


class gaussian(Regress_func):
    def __call__(self, x, mu, sig):
        return np.exp(-((x - mu) / sig) ** 2 / 2)


# %%  Utility regression


def distance_data(h: Houses, i: int, min_dist: float = 0.0005) -> tuple:
    """
    Remove isolated points

    :param min_dist: minimum distance to an other point
    :param h: Houses
    :param i: test index
    :return: var, price
    :rtype: tuple
    """
    var = h(i)
    prices = h(0)
    data = [np.array(e) for e in np.transpose((var, prices)).tolist()]
    data2 = []

    while data:
        pop = False
        k = 1

        while k < len(data):
            if (data[0] - data[k]) @ (data[0] - data[k]) <= min_dist:
                data2.append(data.pop(k))
                pop = True
            else:
                k += 1

        if pop:
            data2.append(data.pop(0))
        else:
            data.pop(0)

    return np.array([e[0] for e in data2]), np.array([e[1] for e in data2])


def test_utility(var: np.ndarray, prices: np.ndarray, min_r2: float = 0.3) -> dict:
    _max = {
        "r2": 0.,
        "f": None,
        "params": None
    }

    for _f in [f for f in Regress_func.__subclasses__() if f.act is True]:
        # noinspection PyBroadException
        try:
            r2, params = fit(_f(), var, prices)
        except Exception:
            pass
        else:
            if r2 > min_r2:
                if r2 > _max["r2"]:
                    _max["r2"] = r2
                    _max["params"] = params
                    _max["f"] = _f

    return _max


class _Utility:
    def __init__(self, label: str, vals: np.ndarray, prices: np.ndarray, dico: dict):
        self.vals = vals
        self.prices = prices
        self.label = label

        def _f(x):
            return dico["f"]()(x, *dico["params"])

        self.func = _f
        self.r2 = dico["r2"]
        self.func_name = dico["f"].__name__

    def __call__(self, x):
        return self.func(x)

    def plot(self) -> plt.Figure:
        x = np.linspace(min(self.vals), max(self.vals))

        fig, ax = plt.subplots()
        ax.plot(self.vals, self.prices, '+')
        ax.plot(x, self(x), '-', label=f"{self.func_name}, R2={str(self.r2)[:4]}")
        ax.set_ylabel("Price")
        ax.set_xlabel(self.label)
        ax.legend()
        return fig


class Utilities:
    def __init__(self, h: Houses, quiet: bool = False):
        self.utilities: list = []
        self.h = h

        li = list(range(len(h.norm_data[0])))[1:]
        for i in li:
            v, p = distance_data(h, i)
            ret = test_utility(v, p)
            lab = h.header_from_int[i]
            if ret["r2"] != 0.:
                if not quiet:
                    print(f"For {lab}: Best function {ret['f'].__name__} with R2={ret['r2']}")
                self.utilities.append(_Utility(lab, v, p, ret))
            else:
                if not quiet:
                    print(f"No function found for {lab}")

    def __len__(self):
        return len(self.utilities)

    def __getitem__(self, index: int):
        return self.utilities[index]

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            ret = self[self.n]
            self.n += 1
            return ret
        else:
            raise StopIteration

    @property
    def data(self) -> Data:
        t = np.array([
            np.array([
                u(e)
                for e in self.h(u.label)
            ])
            for u in self
        ])
        return Data(
            tab=t,
            expected=self.h("price")
        )

    def show_plots(self):
        for u in self:
            u.plot().show()


class HouseLearnerFromUtilities(SimpleNetwork):
    def __init__(self, utilities: Utilities, epochs: int = 10, q: bool = True):
        self.utilities = utilities
        d: Data = self.utilities.data
        super().__init__(
            data=d,
            n_dim=len(self.utilities),
            epochs=epochs,
            quiet=q
        )

# %%  Main functions


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
        print(f"Bluid number {n + 1} over {n_build} :")
        h = Houses("learning_data/kc_house")
        net = HouseLearner(h, epochs=n_epochs)
        _id = int(len(os.listdir(save_folder)) / 3)
        write_json(f"{save_folder}/val_loss{_id}.json", net.history['val_loss'])
        write_json(f"{save_folder}/loss{_id}.json", net.history['loss'])
        write_json(f"{save_folder}/weights{_id}.json", net.weights.tolist())


def build_ut(n_epochs: int = 10, n_build: int = 1, save_folder: str = "data/Obj2/real/default"):
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

    print("Building utilities")
    ut = Utilities(Houses("learning_data/kc_house"))
    print("Utilities built")

    for n in range(n_build):
        print(f"Bluid number {n + 1} over {n_build} :")
        net = HouseLearnerFromUtilities(ut, epochs=n_epochs)
        _id = int(len(os.listdir(save_folder)) / 3)
        write_json(f"{save_folder}/val_loss{_id}.json", net.history['val_loss'])
        write_json(f"{save_folder}/loss{_id}.json", net.history['loss'])
        write_json(f"{save_folder}/weights{_id}.json", net.weights.tolist())


def history_from_file(n: slice = -1, folder: str = "data/Obj2/real/default") -> plt.Figure:
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


# %%  Main


if __name__ == "__main__":
    link = "https://www.kaggle.com/harlfoxem/housesalesprediction"
    print(f"Data from {link}.")

    s = slice(None)
    history_from_file(
        folder="data/Obj2/real/ut1",
        n=s
    ).show()
    weights_from_file(
        folder="data/Obj2/real/ut1",
        n=s,
        labels_link="learning_data/kc_house/header_remove1.json"
    ).show()
