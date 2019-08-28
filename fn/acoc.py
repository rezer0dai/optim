# ANT COLONY OPTIMIZATION for continuous problems
import random
import numpy as np

import matplotlib.pyplot as plt

DEGREE_OF_STEP = .85 # similiar to learning rate, but we dont want to learn fast as noisy and counterproductive
COUNT = 10 # num of ants
EVAPORATION_POOL = COUNT * 12 # archive of trails ( pheromones )
ELITE_POOL = COUNT * 6 # top trails we want to converge

VAR_CERTAINITY = COUNT * 1e-2 # heuristic when model is learning / has converged

from fn import *

def action(mu, sigma):
    a = np.random.normal(mu, sigma)
    if a < VAR_RANGE[0] or a > VAR_RANGE[1]:
        return np.random.uniform(VAR_RANGE[0], VAR_RANGE[1])
    return a

# perf overkill method how is done here, but just for experimenting and clarity of idea
def evaluate(trails, total_cost):
    newest = list(range(len(trails)-COUNT, len(trails)))
    elites = list(total_cost[:-COUNT].argsort()[:COUNT])
    evaporated = [ i for i in range(
        len(trails) - COUNT * 3, # skip our last batch!
        len(trails) - COUNT * 1) if i not in elites ][-COUNT:]
    archive = np.array([
        i for i in range(len(trails)-COUNT) if i not in elites and i not in evaporated ])

#    idx = elites + newest + list(archive) + evaporated
#    return trails[idx], total_cost[idx]

    active = list(archive[total_cost[archive].argsort()[:ELITE_POOL-len(elites)-len(newest)-1*COUNT]])
    passive = [ i for i in archive if i not in active ]

    # BEST | strongest pheromone | best trails | random | weakest pheromone ( oldest ) no significant trails
    idx = elites + newest + active + passive + evaporated
    assert len(idx) == len(trails), "{} vs {}".format(len(idx), len(trails))
    assert len(set(idx)) == len(idx), "{} vs {}".format(len(set(idx)), len(idx))
    return trails[idx], total_cost[idx]

def error(trails, i, mu):
    sigma = np.abs(mu - trails[:ELITE_POOL, len(visited)])
    # heuristic, if 2 possible soutions we dont want to oscilate
    sigma.sort()
    return sigma[1:len(sigma)//2+1].mean()

scores = []
trails = np.random.randn(EVAPORATION_POOL+COUNT, N_VARS)
total_cost = np.array([ cost(trail) for trail in trails ])

print("INITIAL SCORE", np.mean(total_cost))

for i in range(1000):
    var = 0
    for c in range(COUNT):
        score = 0
        visited = []

        stats = []

        trail = trails[random.randint(0, ELITE_POOL-1)]
        while len(visited) != N_VARS:
            mu = trail[len(visited)]
            sigma = DEGREE_OF_STEP * error(trails, len(visited), mu)
            mu_ex = action(mu, sigma)

            var += sigma
            stats.append([mu_ex, mu, sigma, len(visited)])

            visited.append(mu_ex)
            trails[EVAPORATION_POOL + c, len(visited) - 1] = visited[-1]

        score = cost(trails[EVAPORATION_POOL + c])
        total_cost[EVAPORATION_POOL + c] = score

    if var < VAR_CERTAINITY:
        break # heur :: model seems certain enough

    scores.append(np.mean(total_cost[EVAPORATION_POOL:]))
    trails, total_cost = evaluate(trails, total_cost)
    print("EPOCH", i, "cost:", scores[-1], "var", var)

# not too readable as now difference is relative on distance not on order
[print("-->", s) for s in stats]
print("BEST COST:", cost(trails[0]), "solved func :", [int(x+.5) for x in func(trails[0])])
print("FINAL COST:", cost(trails[ELITE_POOL]), "solved func :", func(trails[ELITE_POOL]))

print("SOLUTION:", trails[0])

plt.plot(range(len(scores)), scores)
plt.show()
