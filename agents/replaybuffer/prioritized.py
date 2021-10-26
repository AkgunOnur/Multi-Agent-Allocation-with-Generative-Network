import numpy as np
from collections import namedtuple
from itertools import chain

from dqn.replaybuffer.uniform import BaseBuffer
from dqn.replaybuffer.seg_tree import SumTree, MinTree


class PriorityBuffer(BaseBuffer):
    """ Prioritized Replay Buffer that sample transitions with a probability
    that is proportional to their respected priorities.
        Arguments:
            - capacity: Maximum size of the buffer
            - state_shape: Shape of a single observation (must be a tuple)
            - state_dtype: Data type of the state array. Must be a compatible
            dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype,
                 alpha, epsilon=0.1):
        super().__init__(capacity, state_shape, state_dtype)
        self.sumtree = SumTree(capacity)
        self.mintree = MinTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self._cycle = 0
        self.size = 0

        self.max_p = epsilon ** alpha

        self.transition = namedtuple("Transition", 
                    "state action reward next_state terminal")

    def push(self, transition):
        """ Push a transition object (with single elements) to the buffer.
        Transitions are pushed with the current maximum priority (also push
        priorities to both min and sum tree). Remember to set <_cycle> and
        <size> attributes.
        """

        self._cycle = (self._cycle+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
        
        self.buffer.state[self._cycle] = transition.state
        self.buffer.action[self._cycle] = transition.action
        self.buffer.reward[self._cycle] = transition.reward
        self.buffer.next_state[self._cycle] = transition.next_state
        self.buffer.terminal[self._cycle] = transition.terminal

        self.sumtree.update(self._cycle, self.max_p)
        self.mintree.update(self._cycle, self.max_p)


        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def sample(self, batch_size, beta):
        """ Sample a transition based on priorities.
            Arguments:
                - batch_size: Size of the batch
                - beta: Importance sampling weighting annealing
            Return:
                - batch of samples
                - indexes of the sampled transitions (so that corresponding
                priorities can be updated)
                - Importance sampling weights
        """
        if batch_size > self.size:
            return None
        indice = []
        prios = []
        indxs = []

        segment_size = self.sumtree.tree[0] / batch_size

        for i in range(batch_size):
            a = segment_size * i
            b = segment_size * (i + 1)
            prios.append(np.random.uniform(a, b))

        indxs = np.array([self.sumtree.get(prio) for prio in prios])

        min_p = self.mintree.tree[indxs].min() / self.sumtree.tree.sum()
        max_weight = (min_p * self.size) ** (-beta)

        probability = self.sumtree.tree[indxs] / self.sumtree.tree.sum()
        weights = (self.size * probability) ** (-beta) 
        weights = weights / max_weight

        batch = self.transition(self.buffer.state[indxs],
                    self.buffer.action[indxs],
                    self.buffer.reward[indxs],
                    self.buffer.next_state[indxs],
                    self.buffer.terminal[indxs])

        return batch, indxs, np.array(weights)

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def update_priority(self, indexes, values):
        """ Update the priority values of given indexes (for both min and sum
        trees). Remember to update max_p value! """

        for i in range(len(indexes)):
            self.sumtree.update(indexes[i], values[i] ** self.alpha)
            self.mintree.update(indexes[i], values[i] ** self.alpha)
            self.max_p = max(self.max_p, values[i])

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError
