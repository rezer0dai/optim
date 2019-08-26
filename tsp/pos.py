import random, copy
import numpy as np

import matplotlib.pyplot as plt
def plot_learning(scores):
    _, _ = plt.subplots(1, 1)
    plt.plot(range(len(scores)), scores)
    _, _ = plt.subplots(1, 1)

from tsp import tsp_map, mutate, cost, plot, ind

class Momentum:
    def __init__(self, lr, schedule, momentum):
        self.schedule = schedule
        self.momentum = momentum

        self.vt = 0
        self.lr = lr

    def step(self, dist):
        self.lr = self.lr * self.schedule
        self.vt = self.vt * self.momentum + self.lr * dist
        return 1 + int(self.vt / self.lr)

ELITE = 3

class ParticleLayer:
    def __init__(self, cities, optimizer, temperature):
        out_size = len(cities)
        assert 0 == out_size % 2, "current cost function accepts only odd #cities"

        self.optimizer = optimizer
        self.path = np.random.choice(np.arange(out_size), out_size, replace=False)
        self.error = cost(cities, self.path, *[range(0, len(self.path), 2)]*2)

        self.maping = list(np.argsort(self.path))

        self.temp = temperature

        self.best = None
        self.best = copy.copy(self)
        self.error_degree = None

    def cooldown(self):
        self.temp = max(0, self.temp - 1)
        return self.temp

    def mutate(self, cities, src, dst, cost_src, cost_dst):
        self.maping[self.path[src[0]]] = dst[0]
        self.maping[self.path[src[1]]] = dst[1]

        self.error = self.error - (cost_src - cost_dst)
        self.path[src] = self.path[dst]

#        self.error = cost(cities, self.path, *[range(0, len(self.path), 2)]*2)
#        assert error.round() == self.error.round(), "rel errors {} vs {}".format(
#                error, self.error)

        self._best_update()

    def pclone(self, cities, particle, a):
        target = self.path[a]
        b = particle.maping[target]
#        assert b == list(particle.path).index(target)
        if abs(a - b) < 2 or abs(a - b) == len(self.path) - 1:
            return
        src, dst = [a, b], [b, a]
        cost_src = cost(cities, self.path, src, src)
        cost_dst = cost(cities, self.path, src, dst)
        return self.mutate(cities, src, dst, cost_src, cost_dst)

    def _best_update(self):
        if self.error >= self.best.error:
            return
        self.best = copy.copy(self)

class SwarmNetwork:
    def __init__(self, p_count, n_cities, scale, anealing_degree, lrp, lrg, momentum):
        self.cities = tsp_map(n_cities, scale)

        self.max_error_degree = n_cities# * np.log2(n_cities)
        temperature = int(self.max_error_degree * anealing_degree * np.log2(n_cities))

        self.particles = [
                ParticleLayer(self.cities, Momentum(1e-1, .999, momentum), temperature
                    ) for _ in range(p_count)]

        self.temperature = np.logspace(0, 5, 2 * temperature)[:temperature]
        self.t_i = [len(self.temperature)] * len(self.particles)

        self.lrp = lrp
        self.lrg = lrg

        self.total = 0

    def fit(self, epoch_cap):
        losses = []
        while self.particles[1].temp:#max(self.t_i):
            if len(losses) > epoch_cap:
                break
            loss = self._backprop()
            losses.append(loss)

            temps = [p.temp for p in self.particles]
            print("\nEPOCH", len(losses), self.total, loss, max(temps),
                    "\nSTATS:",
                    "\n", temps,
                    "\n", [p.error_degree for p in self.particles],
                    "\nfirst velo:", self.particles[0].optimizer.vt, self.particles[0].error_degree,
                    "\nlast velo", self.particles[-1].optimizer.vt, self.particles[-1].error_degree)
        print([p.error for p in self.particles])
        print([p.error_degree for p in self.particles])
        return losses

    def _backprop(self):
        self.particles.sort(key=lambda p: p.error)
        # skip outliers
        e_min, e_max = self.particles[3].error, self.particles[-4].error
        for i, p in enumerate(self.particles):
            # velocity like NN momentum
            error = (p.error - e_min)
            # outliers
            if i < 3:
                error = (e_max - e_min) * 0.05
            elif i > len(self.particles)-4:
                error = (e_max - e_min) * .9
            else:
                error = (p.error - e_min)
            if e_max != e_min:
                error /= (e_max - e_min)
            p.error_degree = p.optimizer.step(error * self.max_error_degree)
            error_degree = min(len(p.path)**2, p.error_degree)

            for j in range(error_degree):
                self.total += 1

                # backprop
                idx = random.randint(0, len(p.path)-1)
                if self.lrg > random.random():
                    if self.particles[0].best.path[idx] != p.path[idx]:
                        p.pclone(self.cities, self.particles[random.randint(0, i // 4)], idx)
#                        p.pclone(self.cities, self.particles[random.randint(0, ELITE)], idx)

                idx = random.randint(0, len(p.path)-1)
                if self.lrp > random.random():
                    if p.best.path[idx] != p.path[idx]:
                        p.pclone(self.cities, p.best, idx)

                exporation = 1 > random.randint(0, error_degree)
                dist_proportional = 1 * error < random.random()
                if exporation and dist_proportional:
                    continue # best we want less to improvize

                src = mutate(p.path, degree=2)
                dst = [src[1], src[0]]

                # noise -> out of local minima
                t = self.temperature[p.cooldown()]
                cost_src = cost(self.cities, p.path, src, src)
                cost_dst = cost(self.cities, p.path, src, dst)
                if np.exp((cost_src - cost_dst) / t) > random.random():
                    p.mutate(self.cities, src, dst, cost_src, cost_dst)

        return np.mean([p.error for p in self.particles])

    def plot(self):
        self.particles.sort(key=lambda p: p.error)
        plot(swarm.cities, self.particles[0].best.path)

swarm = SwarmNetwork(10, 24, 1000, 50, lrp=3e-1, lrg=7e-1, momentum=.6)
losses = swarm.fit(300)

plot_learning(losses)
swarm.plot()
