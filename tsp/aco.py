import random
import numpy as np

import matplotlib.pyplot as plt
def plot_learning(scores):
    _, _ = plt.subplots(1, 1)
    plt.plot(range(len(scores)), scores)
    _, _ = plt.subplots(1, 1)

from tsp import tsp_map, plot, ind

COUNT = 100
C_HEUR = 2.5
C_PHERO = 1.
EVAPORATION = .8

cities = tsp_map(24, 1000)

dist = np.array([
    np.sqrt(((cities[a] - cities[b]) ** 2).sum()
        ) if a != b else 1e-8 for b in range(len(cities)) for a in range(len(cities)) ]).reshape(len(cities), -1)

def evaluate(cities, trails, pheromones, total_cost):
    pheromones *= EVAPORATION
    for trail, tc in zip(trails, total_cost):
        for i, c in enumerate(trail):
            pheromones[c, trail[ind(trail, i+1)]] += 1. / tc
            pheromones[trail[ind(trail, i+1)], c] += 1. / tc
    return pheromones

def action(visited, pheromones):
    todos = list( set(range(len(pheromones[0]))) - set(visited) )
    phero = pheromones[ visited[-1] ][todos]
    probs = phero / phero.sum()
    return np.random.choice(todos, size=1, p=probs)[0]

pheromones = np.ones([len(cities), len(cities)]) * 1e-8

scores = []
for i in range(50):
    phero = pheromones.copy()**C_PHERO * (1. / dist)**C_HEUR
    total_cost = []
    trails = []
    for _ in range(COUNT):
        visited = [ random.randint(0, len(cities) - 1) ]
        score = 0
        while len(visited) != len(cities):
            visited.append( action(visited, phero) )
            score += dist[ visited[-1], visited[-2] ]
        total_cost.append(score)
        trails.append(visited)
    pheromones = evaluate(cities, trails, pheromones, total_cost)
    scores.append(np.mean(total_cost))
    print("EPOCH", i, "cost: ", scores[-1])

plot_learning(scores)
plot(cities, trails[np.argsort(total_cost[-len(trails):])[0]])
