import random, copy
import numpy as np

import matplotlib.pyplot as plt

from fn import *

class Momentum:
    def __init__(self, lr, schedule, momentum, n_vars):
        self.schedule = schedule
        self.momentum = momentum

        self.lr = lr.reshape(-1, 1)
        self.vt = np.zeros([lr.shape[0], n_vars])

    def __call__(self, err):
        self.lr = self.lr * self.schedule
        self.vt = self.vt * self.momentum + self.lr * err
        return self.vt

ELITE = 3

class ParticleLayer:
    def __init__(self, n_vars, momentum):
        self.momentum = momentum

        self.path = np.random.randn(n_vars)
        self.error = cost(self.path)
        self.best = copy.copy(self)

    def follow(self, path, velocity):
        delta_soc = path - self.path
        delta_cog = self.best.path - self.path
        # this random.uniform also server purpose of mutation -> no annealing needed here anymore
        delta = np.stack([delta_cog, delta_soc]) * np.random.uniform(0, 1, [2, len(self.path)])

        momentum = self.momentum(delta).sum(0)

        self.path = self.path + momentum * velocity
        self.path[self.path < VAR_RANGE[0]] = VAR_RANGE[0]
        self.path[self.path > VAR_RANGE[1]] = VAR_RANGE[1]

        self._best_update()

    def _best_update(self):
        self.error = cost(self.path)
        if self.error >= self.best.error:
            return
        self.best = copy.copy(self)

class SwarmNetwork:
    def __init__(self, p_count, n_vars, lrp, lrg, momentum):

        self.particles = [
                ParticleLayer(n_vars, Momentum(np.array([lrp, lrg]), .999, momentum, n_vars)
                    ) for _ in range(p_count)]

    def fit(self, epoch_cap):
        losses = []
        for _ in range(epoch_cap):
            loss = self._backprop()
            losses.append(loss)

            print("\nEPOCH", len(losses), loss,
                    "\nSTATS:",
                    "\nfirst velo:", self.particles[0].momentum.vt,
                    "\nlast velo", self.particles[-1].momentum.vt)
        print([p.error for p in self.particles])
        return losses

    def _proportional_distance_velocity(self, e_min, e_max, error, i):
        if i < 3:
            velocity = (e_max - e_min) * 0.05
        elif i > len(self.particles)-4:
            velocity = (e_max - e_min) * .9
        else:
            velocity = (error - e_min)
        if e_max != e_min:
            velocity /= (e_max - e_min)
        velocity = min(.9, max(.05, velocity))
        return velocity

    def _backprop(self):
        self.particles.sort(key=lambda p: p.error)
        # skip outliers
        e_min, e_max = self.particles[3].error, self.particles[-4].error
        for i, p in enumerate(self.particles):
            velocity = self._proportional_distance_velocity(e_min, e_max, p.error, i)
            p.follow(self.particles[0].path, velocity)

        return np.mean([p.error for p in self.particles])

swarm = SwarmNetwork(25, N_VARS, lrp=1, lrg=2, momentum=.5)
losses = swarm.fit(100)

print("BEST COST:", cost(swarm.particles[0].path), "solved func :", [int(x+.5) for x in func(swarm.particles[0].best.path)])
print("SOLUTION:", swarm.particles[0].path)

plt.plot(range(len(losses)), losses)
plt.show()
