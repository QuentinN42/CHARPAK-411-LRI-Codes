import numpy as np
from useful import plot_color as plot, moy, nmap, generate

x = generate()
print(x)
z = np.split(nmap(moy, x), len(x))
print(z)
plot(z, x[:, 0])


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2


def f(tab):
    return sum(tab)


x = np.arange(10)
y = np.arange(10)
z = np.array([np.array([f([_x, _y]) for _x in x]) for _y in y])

fig, ax = plt.subplots()
im = ax.imshow(z)

# We want to show all ticks...
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
# ... and label them with the respective list entries
ax.set_xticklabels(x)
ax.set_yticklabels(y)

fig.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

pcm = plt.pcolormesh(np.random.random((20, 20)), cmap='RdBu_r')
plt.colorbar(pcm)
plt.show()
"""