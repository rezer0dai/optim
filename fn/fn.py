import random
import numpy as np

N_VARS = 10
VAR_RANGE = [-5, 5]

def func(w):
    return (w**2)#.sum()

def cost(w):
    return (np.abs(func(w) - np.arange(len(w)))**2).sum(-1)

def mutation(path, degree=1):
    path = path.copy()
    idx = np.random.choice(
            np.arange(len(path)),
            random.randint(1, degree),
            replace=False)
    noise = np.random.randn(len(idx))

    path[idx] = path[idx] + noise
    cost_m = cost(path)
    return path, cost_m

def scale(path, degree=1):
    path = path.copy()
    idx = np.random.choice(
            np.arange(len(path)),
            random.randint(1, degree),
            replace=False)
#    noise = np.random.randn(len(idx)) # cause big differencies in evolution line ~ family ~ obviousely as changing sign drastically
    noise = np.random.uniform(0, 1, len(idx)) # this will unify ELITE vs FAMILY cost losses direction, but fails to converge

    path[idx] = path[idx] * noise
    cost_m = cost(path)
    return path, cost_m
