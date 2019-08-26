# https://ericphanson.com/blog/2016/the-traveling-salesman-and-10-lines-of-python/
# *also 10 lines of code (w/o comments, empty lines + merge 'if' into one line )
#  but with usage of tsp 'library' :) ~ simulated annealing logic is all here.

import random
import numpy as np

from tsp import tsp_map, mutate, cost, plot

cities = tsp_map(n_cities=20, scale=1000)

path = np.random.choice(np.arange(len(cities)), len(cities), replace=False)
# we cut half of logspace as first/second half is bit brutal for temperature
for temperature in reversed(np.logspace(0, 3, 1e5)[:int(1e5/2)]):
    a, b = mutate(path, degree=2)
    # interesting trick: we dont need to checkpoint best path !!
    # as if new path better np.exp(-..) is low, best path will survive most likely
    if np.exp((
        cost(cities, path, [a, b], [a, b]) - cost(cities, path, [a, b], [b, a])
        ) / temperature) > random.random():

        path[[a, b]] = path[[b, a]]

plot(cities, path)
