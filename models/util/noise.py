import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.num_steps = 0

        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0
            self.c = sigma
            self.sigma_min = sigma

    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
        return sigma

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.num_steps += 1
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class AdaptiveParamNoise:
    def __init__(self, init_std=0.2, desired_std=0.2, alpha=1.01):
        self.init_std = init_std
        self.desired_std = desired_std
        self.alpha = alpha

        self.current_std = init_std

    def adapt(self, distance):
        if distance > self.desired_std:
            # print('down')
            self.current_std /= self.alpha
        else:
            self.current_std *= self.alpha

    def get_std(self):
        return self.current_std