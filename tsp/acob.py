# ANT COLONY OPTIMIZATION BRIDGE ~ continuous space approach for TSP
import random
import numpy as np

import torch
from torch.distributions import Normal

import matplotlib.pyplot as plt
def plot_learning(scores):
    _, _ = plt.subplots(1, 1)
    plt.plot(range(len(scores)), scores)
    _, _ = plt.subplots(1, 1)

from tsp import tsp_map, plot, cost

DEGREE_OF_STEP = .4 # similiar to learning rate, but we dont want to learn fast as noisy and counterproductive
COUNT = 10 # num of ants
EVAPORATION_POOL = COUNT * 12 # archive of trails ( pheromones )
ELITE_POOL = COUNT * 6 # top trails we want to converge

VAR_CERTAINITY = COUNT * 2.5 # heuristic when model is learning / has converged
C_HEUR = 5. # it is depended on DEGREE_OF_STEP

cities = tsp_map(24, 1000)

dist = np.array([
    np.sqrt(((cities[a] - cities[b]) ** 2).sum()
        ) if a != b else 1e-8 for b in range(len(cities)) for a in range(len(cities)) ]).reshape(len(cities), -1)

neighbours = np.vstack([ d.argsort() for i, d in enumerate(dist) ])
diffs = np.vstack([ n.argsort() for i, n in enumerate(neighbours) ])

def random_draw(cities, mu, sigma):
    i = int(abs(np.random.normal(0, sigma)))
    return neighbours[int(mu), int(i)]

def action(cities, visited, mu, sigma):
    if not len(visited):
        return random_draw(cities, mu, sigma)

    locations = neighbours[int(mu)]
    dists = dist[visited[-1], locations]

    # get probs related to archive of ants paths ~ pheromones
    normal = Normal(mu, sigma)
    phero = np.array([ (
        0. if t in visited else max(1e-6, normal.log_prob(i + int(mu)).exp().item())
        ) for i, t in enumerate(locations) ])
    phero = phero / phero.sum()

    # add heuristic for next action
    dists = 1. / dist[visited[-1], locations]**C_HEUR
    dists = dists / dists.sum()

    # calculate next action ~ TSP::city-visit
    probs = phero * 100. * dists
    probs = probs / probs.sum()
    i = np.random.choice(locations, size=1, p=probs)[0]
    assert int(i) not in visited
    return i

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

def error(trails, neighbours, i, mu):
    sigma = np.vstack([
        diffs[int(mu), int(idx)] for idx in trails[:ELITE_POOL, len(visited)] ])
#    for idx in trails[:ELITE_POOL, len(visited)]:
#        assert diffs[int(mu), int(idx)] == list(neighbours[int(mu)]).index(int(idx))

    # heuristic, if 2 possible soutions we dont want to oscilate
    sigma.sort()
    return sigma[1:len(sigma)//2+1].mean()

scores = []
trails = np.vstack([
    np.random.choice(range(len(cities)), size=len(cities), replace=False
        ) for _ in range(EVAPORATION_POOL+COUNT) ])
total_cost = np.array([
    cost(cities, trail, *[range(0, len(trail), 2)]*2) for trail in trails ])

print("INITIAL SCORE", np.mean(total_cost))

for i in range(200):# * EVAPORATION_POOL // COUNT):
    var = 0
    for c in range(COUNT):
        score = 0
        visited = []

        stats = []

        trail = trails[random.randint(0, ELITE_POOL-1)]#EVAPORATION_POOL)]#
        while len(visited) != len(cities):
            mu = trail[len(visited)]
            # not actually same as learning rate,
            # therefore degree of how fierce we want to follow pheromone paths
            sigma = DEGREE_OF_STEP * error(trails, neighbours, len(visited), mu)
            mu_ex = action(cities, visited, mu, sigma)

            var += sigma
            stats.append([mu_ex, mu, sigma, len(visited)])

            visited.append(mu_ex)
            if len(visited) > 1:
                score += dist[ visited[-1], visited[-2] ]
            trails[EVAPORATION_POOL + c, len(visited) - 1] = visited[-1]

        total_cost[EVAPORATION_POOL + c] = score

    if var < VAR_CERTAINITY:
        break # heur :: model seems certain enough

    scores.append(np.mean(total_cost[EVAPORATION_POOL:]))
    trails, total_cost = evaluate(trails, total_cost)
    print("EPOCH", i, "cost:", scores[-1], "var", var)

# not too readable as now difference is relative on distance not on order
[print("-->", s) for s in stats]
plot_learning(scores)
plot(cities, [int(t) for t in trails[0]]) # best
plot(cities, [int(t) for t in trails[COUNT]]) # currently explored
