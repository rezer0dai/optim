# https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/

import random
import numpy as np
import matplotlib.pyplot as plt

SCALE = 1000
N_CITIES = 50

def normalize(cities):
    xmin, xmax = min(cities[:, 0]), max(cities[:, 0])
    ymin, ymax = min(cities[:, 1]), max(cities[:, 1])
    return ((cities - [xmin, ymin]) * 100.) / [xmax-xmin, ymax-xmin]

def ind(path, i):
    if i == -1:
        return len(path) - 1
    if i == len(path):
        return 0
    return i

def c(cities, path, i):
    return cities[ path[ind(path, i)] ]

def cost(cities, path, a, b):
    aa, bb = sorted([a, b])

#    a-1 a a+1 b-1 b b+1
#    a-1 b a+1 b-1 a b+1

    diffs = np.vstack([ 
        ( c(cities, path, j-1) - c(cities, path, i), c(cities, path, i) - c(cities, path, j+1) )
        ] for i, j in zip([a, b], [aa, bb])).reshape(-1, 2)

    return np.sqrt((diffs ** 2).sum(1)).sum()

def mutate(path):
    a, b = sorted(random.sample(range(len(path)), 2))
    if abs(a - b) < 2:
        return mutate(path)
    return a, b

cities = normalize( np.random.randn(N_CITIES, 2) * SCALE )

path = np.random.choice(np.arange(len(cities)), len(cities), replace=False)
for temperature in reversed(np.logspace(0, 3, 1e5)):
    a, b = mutate(path)
    # interesting trick: we dont need to checkpoint best path !!
    # as if new path better np.exp(-..) is low, best path will survive most likely
    if np.exp((cost(cities, path, a, b) - cost(cities, path, b, a)) / temperature) > random.random():
        path[[a, b]] = path[[b, a]]

plt.plot(cities[path][:, 0], cities[path][:, 1], 'xb-')
plt.show()
