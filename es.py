# crossover from core population only
# crossover is core of evolution
# mutation with not big degree, smaller better
#  .. as we want neighbour not too far away
#  .. however 8 degree looks also ok

import random
import numpy as np

import matplotlib.pyplot as plt
def plot_learning(costs):
    plt.plot(range(len(costs)), costs)
    plt.show()

from tsp import tsp_map, mutate, cost, plot

COUNT = 100
MUT_DEGREE = 3#8#
BOUT_SIZE = 7
TOURNAMENT_COUNT = 10
ELITE = 10

cities = tsp_map(24, 1000)

population = [
        np.random.choice(np.arange(len(cities)), len(cities), replace=False
            ) for _ in range(COUNT) ]

fitness = [
        cost(cities, path, *[range(0, len(path), 2)]*2
            ) for path in population ]

def crossover(cities, population, a, b):
    pop_a, pop_b = population[a], population[b]
    a, b = sorted(random.sample(range(len(pop_a)), 2))
#    a = 0

    gene = list(pop_a[a:b])
    pop_c = list(filter(lambda g: g not in gene, pop_b))

    a = random.randint(0, len(pop_c)-1)
    pop_c = np.array(pop_c[:a] + gene + pop_c[a:]).reshape(-1)
    cost_c = cost(cities, pop_c, *[range(0, len(pop_c), 2)]*2)
    return pop_c, cost_c

def mutation(cities, population, a):
    pop_m = population[a].copy()
    idx = mutate(pop_m, degree=random.randint(2, MUT_DEGREE))
    mutation = idx.copy()
    np.random.shuffle(mutation)

    cost_x = cost(cities, pop_m, idx, idx)
    cost_y = cost(cities, pop_m, idx, mutation)

    cost_m = fitness[a] - (cost_x - cost_y)
    pop_m[idx] = pop_m[mutation]
#    cost_w = cost(cities, pop_m, *[range(0, len(pop_m), 2)]*2)
#    print(cost_m, cost_w)
#    assert cost_m.round() == cost_w.round()
    return pop_m, cost_m

def fittest(elite, population, fitness):
    a = elite[random.randint(0, len(elite) - 1)]
    return population[a], fitness[a]

def evaluate(fitness, a):
    score = 0
    for _ in range(BOUT_SIZE):
        score += fitness[a] < fitness[random.randint(0, len(fitness)-1)]
    return score

def tournament(fitness):
    idx = np.random.choice(range(len(fitness)), TOURNAMENT_COUNT, replace=False)
    f = np.asarray(fitness)
    i = f[idx].argsort()[:2]
    return idx[i]

scores = []
for _ in range(200):
    elite = np.argsort(fitness)[:ELITE]
    for i in range(COUNT):
#        out = '''
        op = random.randint(0, 2)
        a, b = tournament(fitness)
        if 1 > op:
            pop_c, fit_c = mutation(cities, population, a)
        elif 3 > op:
            pop_c, fit_c = crossover(cities, population, a, b)
        elif 3 == op:
            pop_c, fit_c = fittest(population, fitness)
        fitness.append(fit_c)
        population.append(pop_c)

    elite = np.argsort(fitness)
    breed = np.argsort([evaluate(fitness, i) for i in range(len(fitness))])
    evolution = np.concatenate([elite[:ELITE], breed[:COUNT-ELITE]])

    scores.append(np.asarray(fitness)[evolution].mean())#elite[:ELITE]].mean())

    fitness = [ fitness[i] for i in evolution ]
    population = [ population[i] for i in evolution ]

plot(cities, population[np.argsort(fitness)[0]])
plot_learning(scores)
