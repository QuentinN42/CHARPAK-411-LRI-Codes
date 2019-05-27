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


#%%  Class Houses


class Houses:
    def __init__(self, link):
        self.header_from_int = {i: e for i, e in enumerate(get_json(link + '/header.json')[2:])}
        self.header_from_str = {v: k for k, v in self.header_from_int.items()}

        self.raw_data: np.ndarray = np.genfromtxt(link + '/raw.csv', delimiter=',')[:, 2:]
        for i in range(len(self.raw_data[:, self.header_from_str["yr_renovated"]])):
            if self.raw_data[i, self.header_from_str["yr_renovated"]] == 0:
                self.raw_data[i, self.header_from_str["yr_renovated"]] = \
                    self.raw_data[i, self.header_from_str["yr_built"]]

        self.norm_data: np.ndarray = self.raw_data / np.linalg.norm(self.raw_data)
        """
        np.array([
            np.array([
                (self.raw_data[j, i] - min(self.raw_data[:, i]))/(max(self.raw_data[:, i]) - min(self.raw_data[:, i]))
                for j in range(len(self[i]))
            ])
            for i in range(len(self.raw_data))
        ])
        """

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

    def color_scatter(self, c, norm: bool = False) -> plt.figure:
        fig, ax = plt.subplots()
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


def build(n_epochs: int = 10, n_build: int = 1):
    for n in range(n_build):
        print(f"Bluid number {n+1} over {n_build} :")
        h = Houses("learning_data/kc_house")
        net = HouseLearner(h, epochs=n_epochs)
        _id = int(len(os.listdir('data/Obj2/real'))/3)
        write_json(f"data/Obj2/real/val_loss{_id}.json", net.history['val_loss'])
        write_json(f"data/Obj2/real/loss{_id}.json", net.history['loss'])
        write_json(f"data/Obj2/real/weights{_id}.json", net.weights.tolist())


def history_from_file(n: int = -1) -> plt.Figure:
    """
    Plot form data/Obj2/real/...n.json
    :param n: data index if -1 the last index is returned
    :return:
    """
    _id = list(range(int(len(os.listdir('data/Obj2/real')) / 3)))[n]
    dico = {
        "loss": get_json(f"data/Obj2/real/loss{_id}.json"),
        "val_loss": get_json(f"data/Obj2/real/val_loss{_id}.json")
    }
    return history_plot(dico, "loss", True)


def weights_from_file(n: list = -1, labels_link: str = None) -> plt.Figure:
    """
    Plot form data/Obj2/real/...n.json
    :param labels_link: link to show labels
    :param n: data index if -1 the last index is returned
    :return:
    """
    fig, ax = plt.subplots()
    if type(n) is int:
        _id = list(range(int(len(os.listdir('data/Obj2/real')) / 3)))[n]
        weights = get_json(f"data/Obj2/real/weights{_id}.json")
        list_n = list(range(len(weights)))
        norm_weights = [e / len(weights) for e in weights]
        norm_weights_a = [abs(e) for e in norm_weights]
        ax.bar(
            list_n,
            norm_weights_a,
            color=['red' if e < 0 else 'blue' for e in norm_weights]
        )
    elif type(n) in [list, np.ndarray, tuple]:
        _ids = [list(range(int(len(os.listdir('data/Obj2/real')) / 3)))[i] for i in n]
        weightss = my_zip([get_json(f"data/Obj2/real/weights{_id}.json") for _id in _ids])
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


#%%  Main


if __name__ == "__main__":
    link = "https://www.kaggle.com/harlfoxem/housesalesprediction"
    print(f"Data from {link}.")
    build(n_epochs=100)
    weights_from_file(
        list(range(5, 15)),
        labels_link="learning_data/kc_house/header.json"
    ).show()
