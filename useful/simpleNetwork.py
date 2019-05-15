"""
A simple Dense network.

@date: 15/05/2019
@author: Quentin Lieumont
"""
from useful.network import Network
from useful.data import Data
from useful.functions import generate
from keras import layers


class SimpleNetwork(Network):
    """
    Input : X n dim array
    Output : activation(X.W)
    with W the weights vector
    """
    def __init__(self,
                 # Data initialisation
                 data: Data = None,
                 func: callable = None,
                 n_dim: int = 2,
                 
                 # Layer options
                 use_bias: bool = False,
                 activation: str = 'linear',
                 
                 # Training options
                 split_ratio: float = 0.5,
                 validate: bool = True,
                 plot_history: bool = False,
                 plot_acc: bool = False,
                 plot_loss: bool = False,
                 save_link: str = ""
                 ):
        self.n_dim = n_dim
        if func:
            if not data:
                d = generate(dim=self.n_dim)
                data = Data(d, func)
            else:
                data = Data(data.raw_data, func)
        else:
            if not data:
                raise AttributeError('Data generation need function')
        super().__init__(data)
        self.model.add(layers.Dense(1, activation=activation, input_dim=self.n_dim, use_bias=use_bias))
        self.build()
        self.train(split_ratio, validate, plot_history, plot_acc, plot_loss, save_link)


if __name__ == '__main__':
    net = SimpleNetwork(func=sum)
    net.graph_color()
