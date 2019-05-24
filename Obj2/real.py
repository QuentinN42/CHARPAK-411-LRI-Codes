"""
Real dataset from https://www.kaggle.com/harlfoxem/housesalesprediction

@date: 23/05/2019
@author: Quentin Lieumont
"""
import numpy as np
from matplotlib import pyplot as plt
from useful.functions import get_json, write_json
from useful.simpleNetwork import SimpleNetwork
from useful.data import Data
import os


#%%  Class Houses


class Houses:
    def __init__(self, link):
        self.header_from_int = {i: e for i, e in enumerate(get_json(link + '/header.json')[2:])}
        self.header_from_str = {v: k for k, v in self.header_from_int.items()}

        self.raw_data: np.ndarray = np.genfromtxt(link + '/raw.csv', delimiter=',')[:, 2:]

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
    def __init__(self, houses: Houses, epochs: int = 10):
        self.houses = houses
        d: Data = Data(tab=houses.norm_data[:, 1:], expected=houses("price"))
        super().__init__(
            data=d,
            n_dim=len(houses.norm_data[0, 1:]),
            epochs=epochs,
            quiet=False
        )


#%%  Main


def build():
    h = Houses("learning_data/kc_house")
    net = HouseLearner(h)
    # net.graph_history('loss')
    _id = int(len(os.listdir('data/Obj2/real'))/2)
    print(_id)
    write_json(f"data/Obj2/real/val_loss{_id}.json", net.history['val_loss'])
    write_json(f"data/Obj2/real/loss{_id}.json", net.history['loss'])


if __name__ == "__main__":
    link = "https://www.kaggle.com/harlfoxem/housesalesprediction"
    print(f"Data from {link}.")
    build()
