import numpy as np

N_VARS = 10
VAR_RANGE = [-5, 5]

def func(w):
    return (w**2)#.sum()

def cost(w):
    return (np.abs(func(w) - np.arange(len(w)))**2).sum()
