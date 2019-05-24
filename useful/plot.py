"""
Plot data

@date: 22/05/2019
@author: Quentin Lieumont
"""
from matplotlib import pyplot as plt
from useful.functions import get_json, idem_items, average, std_err


class One_Line:
    """
    One line of data
    """

    header: dict
    result: list
    res_key: str = "result"
    bool_labels = ["Random", "Sorted"]

    def __init__(self, raw_line: dict):
        self.header = {k: raw_line[k] for k in raw_line.keys() if k != self.res_key}
        self.result = raw_line[self.res_key]

    def __len__(self):
        return len(self.result)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return self.header == other.header

    def __getitem__(self, item):
        return self.header[item]

    def __repr__(self):
        b = str(self.bool_labels[int(self.header["sort"])][0])
        s = str(self.header["size"])
        n = str(self.header["n"])
        f = " " + str(self.header["loss_f"])
        return "".join([s, b, n, f])

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        if other != self:
            raise ArithmeticError(f"+ is only between to One_Line, got {type(other)}")
        tmp = self
        tmp.result += other.result
        return One_Line(tmp.dict)

    def same_label(self, other) -> bool:
        return self.plot_label == other.plot_label

    @property
    def plot_label(self) -> str:
        b = str(self.bool_labels[int(self.header["sort"])])
        f = str(self.header["loss_f"])
        return f"{f} {b}"

    @property
    def dict(self) -> dict:
        tmp = self
        tmp.header.update({tmp.res_key: tmp.result})
        return tmp.header

    @property
    def err_tab(self) -> list:
        tab = []
        for i in range(len(self.result[0])):
            tab.append(std_err([e[i] for e in self.result]))
        return tab

    @property
    def average(self) -> float:
        return average(self.err_tab)

    @property
    def std_err(self):
        return std_err(self.err_tab)


class To_Plot_Data:
    """
    All datas to plot
    """

    data: list
    xlabel: dict = {
        "n": "Number of learning, Size of the dataset: {s} values",
        "size": "Size of the dataset, Number of learning: {n} learning",
    }
    colors = ["b", "c", "r", "m", "y", "g", "k", "k", "k", "k"]

    def __init__(self, data):
        if type(data) is str:
            dicos = [e for e in get_json(data) if e != {}]
        elif type(data) is list:
            dicos = data
        else:
            raise AttributeError("Data must be init with str or list type")
        self.data = []
        for d in dicos:
            line = One_Line(d)
            if line not in self.data:
                self.data.append(line)
            else:
                self.data[self.data.index(line)] += line

    def print_all(self):
        for line in self.data:
            print(str(line) + f" ==> {len(line)}")

    def __add__(self, other):
        return To_Plot_Data(self.dict + other.dict)

    @property
    def dict(self):
        return [d.dict for d in self.data]

    @property
    def variations(self):
        if not idem_items(self.ns):
            key = "n"
            if not idem_items(self.sizes):
                raise AttributeError("Variation over 2 axis")
        elif not idem_items(self.sizes):
            key = "size"
        else:
            raise AttributeError("No variation detected")
        return key

    def with_same_label(self, test: One_Line):
        return [l for l in self.data if l.same_label(test)]

    @property
    def to_plot(self):
        li = []
        for l in self.data:
            if self.with_same_label(l) not in li:
                li.append(self.with_same_label(l))
        return li

    def plot(self):
        fig, ax = plt.subplots()
        # ax.set_ylim(0)
        ax.set_xlabel(
            self.xlabel[self.variations].format(
                n=self.data[0]["n"], s=self.data[0]["size"]
            )
        )
        for i, tab in enumerate(self.to_plot):
            x = [l[self.variations] for l in tab]
            err = [l.std_err for l in tab]
            ave = [l.average for l in tab]
            aveh = [ave[i] + err[i] for i in range(len(tab))]
            aveb = [ave[i] - err[i] for i in range(len(tab))]

            c = self.colors[i]

            ax.plot(x, ave, "+" + c, label=tab[0].plot_label)
            ax.plot(x, aveh, "v-" + c)
            ax.plot(x, aveb, "^-" + c)

        ax.legend()
        fig.show()

    @property
    def ns(self):
        return [e["n"] for e in self.data]

    @property
    def sizes(self):
        return [e["size"] for e in self.data]

    @property
    def funcs(self):
        return [e["loss_f"] for e in self.data]

    @property
    def sorts(self):
        return [e["sort"] for e in self.data]


if __name__ == "__main__":
    data = To_Plot_Data("data/json/test_n2.json")
    data.print_all()
    data.plot()
