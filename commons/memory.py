import random

import numpy as np

from commons.tree import SumSegmentTree, MinSegmentTree, SumTree


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._kind = 'simple'

    def __len__(self):
        return len(self._storage)

    def add(self, step):
        if self._next_idx >= len(self._storage):
            self._storage.append(step)
        else:
            self._storage[self._next_idx] = step
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, beta = None):
        idxs = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxs), (None, None)

    def update_memory(self, idxs, ws):
        raise NotImplementedError()


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        # alpha define how much prioritization is used
        super(PrioritizedReplayBuffer, self).__init__(size=size)
        assert alpha > 0
        self._alpha = alpha
        self._kind = 'prop_replay'
        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2
        self._sum_tree = SumSegmentTree(capacity=tree_capacity)
        self._min_tree = MinSegmentTree(capacity=tree_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._sum_tree[idx] = self._max_priority ** self._alpha
        self._min_tree[idx] = self._max_priority ** self._alpha

    def _sample_prop(self, batch_size):
        idxs = []
        for _ in range(batch_size):
            prob = random.random() * self._sum_tree.sum(0, len(self._storage) - 1)
            idx = self._sum_tree.find_prefixsum_idx(prob)
            idxs.append(idx)
        return idxs

    def sample(self, batch_size, beta=None):
        assert beta > 0
        # degree to use the importance weight
        # 0 no correction, 1 full corretion
        idxs = self._sample_prop(batch_size)
        ws = []
        p_min = self._min_tree.min() / self._sum_tree.sum()
        max_w = (p_min * len(self._storage) ** (-beta))

        for idx in idxs:
            p_sample = self._sum_tree[idx] / self._sum_tree.sum()
            w = (p_sample * len(self._storage) ** (-beta))
            ws.append(w / max_w)
        ws = np.array(ws)
        sample = self._encode_sample(idxs)
        return sample, (idxs, ws)

    def update_memory(self, idxs, ws):
        assert len(idxs) == len(ws)
        for idx, w in zip(idxs, ws):
            assert w > 0
            assert 0 <= idx < len(self._storage)
            self._sum_tree[idx] = w ** self._alpha
            self._min_tree[idx] = w ** self._alpha
            self._max_priority = max(self._max_priority, w)


class Experience(object):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with
    probability in proportion to sample's priority, update
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """

    def __init__(self, memory_size, batch_size, alpha=0.6):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha

    def add(self, data, priority):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority ** self.alpha)

    def select(self, beta=0.4):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0])  # To avoid duplicating

        self.priority_update(indices, priorities)  # Revert priorities
        weights = np.array(weights)
        weights /= weights.max()  # Normalize for stability

        out = self.prepare_output(out)
        return out, weights, indices

    def prepare_output(self, out):
        batch = np.array(out)

        s1_batch = np.vstack(batch[:, 0])
        a_batch = np.vstack(batch[:, 1])
        r_batch = batch[:, 2]
        s2_batch = np.vstack(batch[:, 3])
        t_batch = batch[:, 4]

        return s1_batch, a_batch, r_batch, s2_batch, t_batch

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

