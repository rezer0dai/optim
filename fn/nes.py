# EXPERIMENTAL SETUP with goals : NES as multilayered-network + trying to combine with Ant Colony opt, to try on TSP later

import numpy as np

class SGD:
    def __init__(self, lr, schedule):
        self.lr = lr
        self.schedule = schedule

    def step(self, w, grad, batch_size):
        self.lr = self.lr * self.schedule
        return w - grad * self.lr / batch_size

class NAG:
    def __init__(self, lr, schedule, momentum):
        self.schedule = schedule
        self.momentum = momentum

        self.vt = 0
        self.lr = lr

    def step(self, w, grad, batch_size):
        self.lr = self.lr * self.schedule
        w = w + self.vt * self.lr * self.momentum
        self.vt = self.vt * self.momentum + grad / batch_size
        w = w - self.vt * self.lr
        return w - self.vt * self.lr * self.momentum

class NoisyLinear:
    def __init__(self, in_dim, out_dim, batch_size, sigma, optim):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(1. / out_dim)
        self.noise = None

        self.sigma = sigma
        self.optim = optim
        self.population_size = batch_size

    def fire(self, output): # ant colony idea; perf overkill but want to experiment a little
        signal = np.zeros([output.shape[0], output.shape[2]])
        for k, o in enumerate(output):
            for i in range(N_VARS):
                s = np.abs(o[k, i] - o[:, i])
                s.sort()
                sigma = s[1:len(s)//2+1].mean()
                signal[k, i] = np.random.normal(o[k, i], sigma)
        return signal

    # EXPERIMENTAL SETUP, one layer all noises; best to separate for multiple network where layer have only one noise
    # here we do for convinience, and experimenting with fire(..)
    def forward(self, data):
        self.noise = np.random.randn(self.population_size, *self.w.shape) if self.noise is None else self.noise
        output = data.dot(self.w + self.noise * self.sigma)
        output = output[ (list(range(len(output))), list(range(len(output)))) ]
#        output = self.fire(output)
        return output

    def backward(self, error, mask):
        self.grad = (self.noise * mask).T.dot(error).T
        return np.ones([self.noise.shape[0], 1, self.noise.shape[1]])

    def step(self):
        self.noise = None
        self.w = self.optim.step(self.w, self.grad, self.population_size * self.sigma)

class ReLU:
    def forward(self, x):
        self.out = x >= 0.
        return x * self.out

    def backward(self, error, mask):
#        return np.ones([self.out.shape[0], 1, self.out.shape[1]])
        return self.out.reshape(self.out.shape[0], 1, self.out.shape[1])

    def step(self):
        pass

class Network:
    def __init__(self, layers, loss):
        self.loss = loss
        self.layers = layers

    def predict(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def fit(self, x):
        y_pred = self.predict(x)

        error = self.loss(y_pred)
        e_norm = self._normalize(error)

        mask = np.ones(self.layers[-1].noise.shape)
        for l in reversed(self.layers):
            mask = l.backward(e_norm, mask)

        for l in self.layers:
            l.step()

        return error.mean()

    def _normalize(self, e):
        return (e - e.mean()) / (1e-8 + e.std())

from fn import *
import matplotlib.pyplot as plt

batch_size = 64
#solution = np.vstack([ (i + np.arange(N_VARS).reshape(1, N_VARS)) / 10. for i in range(batch_size)])
solution = np.vstack([ (3 + np.arange(N_VARS).reshape(1, N_VARS)) / 1. for i in range(batch_size)])
def cost_ex(w):
    return (np.abs(func(w) - solution)**2).sum(-1)

lr = 1e-3
gamma = 1 - 1e-2
ff = Network([
#        NoisyLinear(1, N_VARS, batch_size=batch_size, sigma=1e-2, optim=SGD(lr, gamma)),
        NoisyLinear(1, N_VARS, batch_size=batch_size, sigma=1e-2, optim=NAG(lr, gamma, .9)),

#        NoisyLinear(1, 16, batch_size=batch_size, sigma=1e-2, optim=SGD(lr, gamma)),#NAG(lr, gamma, .9)),#
#        ReLU(),
##        NoisyLinear(16, 16, batch_size=batch_size, sigma=1e-2, optim=NAG(lr, gamma, .9)),#SGD(lr, gamma)),#
##        ReLU(),
#        NoisyLinear(16, N_VARS, batch_size=batch_size, sigma=1e-1, optim=NAG(lr, gamma, .9)),#SGD(lr, gamma)),#
    ], loss=cost_ex)

scores = []
for _ in range(300):
    e = ff.fit(np.ones([batch_size, 1]))
    scores.append(e.item())

path = ff.predict(np.ones([64, 1]))[0]
print("SOLUTION:", [int(x+.5) for x in func(path)])
plt.plot(range(len(scores)), scores)
plt.show()
