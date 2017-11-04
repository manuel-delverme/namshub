import random

import numpy as np


# TODO replace with code from dyn walk and use prioritized memory
class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def add(self, step):
        if self._next_idx >= len(self._storage):
            self._storage.append(step)
        else:
            self._storage[self._next_idx] = step
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _prepare_data(self, idxs):
        obs, acts, rws, obs1, dones = [], [], [], [], []
        for i in idxs:
            step = self._storage[i]
            ob, act, r, ob1, done = step
            obs.append(np.array(ob, copy=False))
            acts.append(np.array(act, copy=False))
            rws.append(r)
            obs1.append(np.array(ob1, copy=False))
            dones.append(done)
        return np.array(obs), np.array(acts), np.array(rws), np.array(obs1), np.array(dones)

    def sample(self, batch_size):
        idxs = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._prepare_data(idxs=idxs)
