"""
Experience replay buffer for DQN training.

Implements a simple circular buffer that stores transitions and
supports random sampling for off-policy learning.
"""

import random
from collections import deque, namedtuple

import numpy as np


Transition = namedtuple("Transition", "s a r s2 d")


class Replay:
    """
    Fixed-size experience replay buffer.

    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training. Uses a circular buffer that
    automatically discards oldest transitions when capacity is reached.

    Args:
        cap: Maximum number of transitions to store.
    """

    def __init__(self, cap=50_000):
        self.buf = deque(maxlen=cap)

    def push(self, *args):
        """
        Add a new transition to the buffer.

        Args:
            *args: Unpacked transition tuple (state, action, reward, next_state, done).
        """
        self.buf.append(Transition(*args))

    def sample(self, bs=32):
        """
        Sample a random batch of transitions.

        Args:
            bs: Batch size to sample.

        Returns:
            Tuple of numpy arrays (states, actions, rewards, next_states, dones).
            Each array has batch_size as the first dimension.
        """
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return len(self.buf)


class PrioritizedReplay:
    """
    Prioritized Experience Replay buffer.

    Samples transitions with probability proportional to their TD error.
    Implements importance sampling weights to correct for bias.

    Based on "Prioritized Experience Replay" (Schaul et al., 2015).

    Args:
        cap: Maximum number of transitions to store.
        alpha: Prioritization exponent (0 = uniform, 1 = full prioritization).
        beta_start: Initial importance sampling weight exponent.
        beta_frames: Number of frames to anneal beta to 1.0.
    """

    def __init__(self, cap=50_000, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.cap = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buf = []
        self.pos = 0
        self.priorities = np.zeros((cap,), dtype=np.float32)

    def push(self, *args):
        """
        Add a new transition to the buffer with maximum priority.

        Args:
            *args: Unpacked transition tuple (state, action, reward, next_state, done).
        """
        max_prio = self.priorities.max() if self.buf else 1.0

        if len(self.buf) < self.cap:
            self.buf.append(Transition(*args))
        else:
            self.buf[self.pos] = Transition(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.cap

    def sample(self, bs=32):
        """
        Sample a batch of transitions according to priorities.

        Args:
            bs: Batch size to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights).
            - indices: Indices of sampled transitions for priority updates.
            - weights: Importance sampling weights.
        """
        N = len(self.buf)
        if N == self.cap:
            prios = self.priorities
        else:
            prios = self.priorities[:N]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(N, bs, p=probs)
        samples = [self.buf[idx] for idx in indices]

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        s, a, r, s2, d = map(np.stack, zip(*samples))
        return s, a, r, s2, d, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update.
            priorities: New priority values (typically absolute TD errors).
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return len(self.buf)
