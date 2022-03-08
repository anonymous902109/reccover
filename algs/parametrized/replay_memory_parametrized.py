import random
from collections import namedtuple
import numpy as np
from itertools import chain


ParametrizedTransition = namedtuple('ParametrizedTransition',
                                    ('state', 'action', 'next_state', 'reward', 'not_done', 'mask'))


class ReplayMemoryParametrized(object):

    def __init__(self, capacity, num_masks):
        self.capacity = capacity
        self.memories = [[] for i in range(num_masks)]
        self.counts = np.zeros((num_masks, ), dtype=int)
        self.positions = np.zeros((num_masks, ), dtype=int)

    def push(self, *args):
        """Saves a transition."""
        transition = ParametrizedTransition(*args)
        mask_id = self.encode(transition[-1])

        if self.counts[mask_id] < self.capacity:
            self.memories[mask_id].append(None)

        next_pos = self.positions[mask_id]

        self.memories[mask_id][next_pos] = transition
        self.positions[mask_id] = (self.positions[mask_id] + 1) % self.capacity
        self.counts[mask_id] += 1

    def sample(self, batch_size, mask=None):
        if mask is None:  # return a random sample from a flattened list
            return random.sample(list(chain(*self.memories)), batch_size)

        # otherwise, return a sample with specified mask
        mask_id = self.encode(mask)
        try:
            return random.sample(self.memories[mask_id], batch_size)
        except:
            return random.sample(list(chain(*self.memories)), batch_size)

    def __len__(self):
        return np.sum(self.counts)

    def encode(self, binary): #  TODO: refactor this
        binary = binary.cpu()
        binary = binary.squeeze().numpy()
        decimal = 0
        deg = 0
        for i in reversed(binary):
            if deg == 0:
                decimal += i**2
            else:
                decimal += (i*2) ** deg

            deg += 1

        return int(decimal)



