import numpy as np
from collections import deque


class HistoryBuffer(object):
    def __init__(self, preprocess_fn, state_shape, stacked_frames_for_state):
        self.buf = deque(maxlen=stacked_frames_for_state)
        self.preprocess_fn = preprocess_fn
        self.state_shape = state_shape
        self.clear()

    def clear(self):
        for i in range(self.buf.maxlen):
            self.buf.append(np.zeros(self.state_shape, np.float32))

    def add(self, o):
        self.buf.append(self.preprocess_fn(o))
        state = np.concatenate([s_i for s_i in self.buf], axis=2)
        return state
