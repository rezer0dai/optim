import random, copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fn import *

COUNT = 100
BOUT_SIZE = 7
TOURNAMENT_COUNT = 10
ELITE = 10

population = list(np.random.randn(COUNT, N_VARS))
fitness = [ cost(pop) for pop in population ]

def crossover(population, a, b):
    pop_a, pop_b = population[a], population[b]
    a, b = sorted(random.sample(range(len(pop_a)), 2))

    gene = pop_a[a:b]

    pop_c = np.hstack([pop_b[:a], gene, pop_b[b:]])

    cost_c = cost(pop_c)
    return pop_c, cost_c

def pclone(population, a, b):
    pop_c = copy.copy(population[a])
    i = random.randint(0, len(pop_c)-1)

    pop_c[i] = population[b][i]
    cost_c = cost(pop_c)
    return pop_c, cost_c

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

scores_family = []
scores_fitest = []
for _ in range(50):
    elite = np.argsort(fitness)[:ELITE]
    for i in range(COUNT):
        op = random.randint(0, 3)
        a, b = tournament(fitness)
        if 1 > op:
            pop_c, fit_c = mutation(population[a], N_VARS)
        elif 2 > op:
            pop_c, fit_c = scale(population[a], N_VARS)
        elif 3 > op:
            pop_c, fit_c = pclone(population, a, b)
        elif 4 > op:
            pop_c, fit_c = crossover(population, a, b)
        fitness.append(fit_c)
        population.append(pop_c)

    elite = np.argsort(fitness)
    breed = np.argsort([evaluate(fitness, i) for i in range(len(fitness))])
    evolution = np.concatenate([elite[:ELITE], breed[:COUNT-ELITE]])

    scores_family.append(np.asarray(fitness)[evolution].mean())
    scores_fitest.append(np.asarray(fitness)[elite[:ELITE]].mean())

    fitness = [ fitness[i] for i in evolution ]
    population = [ population[i] for i in evolution ]

print("BEST COST:", cost(population[0]), "solved func :", [int(x+.5) for x in func(population[0])])
print("SOLUTION:", population[0])

plt.ylabel("losses")
plt.xlabel("#training steps")
family = mpatches.Patch(color="red", label="family losses")
fitest = mpatches.Patch(color="green", label="fittest losses")
plt.legend(handles=[family, fitest])
#plt.plot(range(len(scores_family)), max(scores_fitest) + np.asarray(scores_family) / 10., color="red")
plt.plot(range(len(scores_family)), scores_family, color="red")
plt.plot(range(len(scores_fitest)), scores_fitest, color="green")
plt.show()
