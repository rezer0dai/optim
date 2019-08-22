import random
import numpy as np
import matplotlib.pyplot as plt

def normalize(cities):
    xmin, xmax = min(cities[:, 0]), max(cities[:, 0])
    ymin, ymax = min(cities[:, 1]), max(cities[:, 1])
    return ((cities - [xmin, ymin]) * 100.) / [xmax-xmin, ymax-xmin]

def tsp_map(n_cities, scale):
    return normalize( np.random.randn(n_cities, 2) * scale )

def ind(path, i):
    if i == -1:
        return len(path) - 1
    if i == len(path):
        return 0
    return i

def c(cities, path, i):
    return cities[ path[ind(path, i)] ]

# cost we calculate only diff between two paths, no need cost of full path ..
def cost(cities, path, a, b):
#    a-1 a a+1 b-1 b b+1
#    a-1 b a+1 b-1 a b+1
    diffs = np.vstack([
        ( c(cities, path, j-1) - c(cities, path, i), c(cities, path, i) - c(cities, path, j+1) )
        ] for j, i in zip(a, b)).reshape(-1, 2)

    return np.sqrt((diffs ** 2).sum(1)).sum()

def mutate(path, degree):
    a, b = (1, 0) if random.randint(0, 1) else (0, 1)
    mutation = np.random.choice(np.arange(a, len(path)-b), degree, replace=False)
    mutation.sort()
    if np.any(np.abs(mutation[:-1] - mutation[1:]) < 2):
        return mutate(path, degree)
    return mutation

def plot(cities, path):
    plt.plot(cities[path][:, 0], cities[path][:, 1], 'xb-')
    plt.show()
