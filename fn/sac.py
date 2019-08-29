import random
import numpy as np

import matplotlib.pyplot as plt

from fn import *

scores = []
path = np.random.randn(N_VARS)
EPOCH_CAP = int(1e5)
for temperature in reversed(np.logspace(0, 4, EPOCH_CAP * 2)[:EPOCH_CAP]):
    p, c = mutation(path, N_VARS // 2)
    if np.exp(( cost(path) - c ) / temperature) < random.random():
        continue
    path[:] = p
    scores.append(c)

print("BEST COST:", cost(path), "solved func :", [int(x+.5) for x in func(path)])
print("SOLUTION:", path)

plt.plot(range(len(scores)), scores)
plt.show()
