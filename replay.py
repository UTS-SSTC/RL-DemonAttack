import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple("Transition", "s a r s2 d")


class Replay:
    def __init__(self, cap=50_000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, bs=32):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)
