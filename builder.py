from useful.data import Data
from useful.choquet import Choquet
import numpy as np

v1 = np.array([0.5, 0.5, 0.8])
v2 = np.array([0.1, 0.5, 0.2])
v3 = np.array([0.2, 0.1, 0.5])
ch = Choquet(v1, v2, v3)
chd = Data(func=ch, debug=True)

chd.save('data/data.json')
