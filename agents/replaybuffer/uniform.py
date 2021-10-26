""" Vanilla Replay Buffer
"""
import numpy as np
from collections import namedtuple


class BaseBuffer():
    """ Base class for 1-step buffers. Numpy queue implementation with
    multiple arrays. Sampling efficient in numpy (thanks to fast indexing)

    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the sumtreestate array. Must be a
        compatible type to numpy
    """

    Transition = namedtuple("Transition",
                            "state action reward next_state terminal")

    def __init__(self, capacity, state_shape, state_dtype):

        self.capacity = capacity

        if not isinstance(state_shape, (tuple, list)):
            raise ValueError("State shape must be a list or a tuple")

        self.transition_info = self.Transition(
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.int64},
            {"shape": (1,), "dtype": np.float32},
            {"shape": state_shape, "dtype": state_dtype},
            {"shape": (1,), "dtype": np.float32},
        )

        self.buffer = self.Transition(
            *(np.zeros((capacity, *x["shape"]), dtype=x["dtype"])
              for x in self.transition_info)
        )

    def __len__(self):
        """ Capacity of the buffer
        """
        return self.capacity

    def push(self, transition, *args, **kwargs):
        """ Push a transition object (with single elements) to the buffer
        """
        raise NotImplementedError

    def sample(self, batchsize, *args, **kwargs):
        """ Sample a batch of transitions
        """
        raise NotImplementedError


class UniformBuffer(BaseBuffer):
    """ Standard Replay Buffer that uniformly samples the transitions.
    Arguments:
        - capacity: Maximum size of the buffer
        - state_shape: Shape of a single observation (must be a tuple)
        - state_dtype: Data type of the state array. Must be a compatible
        dtype to numpy
    """

    def __init__(self, capacity, state_shape, state_dtype):
        super().__init__(capacity, state_shape, state_dtype)
        self._cycle = 0
        self.size = 0

    def push(self, transition):
        """ Push a transition object (with single elements) to the buffer.
        FIFO implementation using <_cycle>. <_cycle> keeps track of the next
        available index to write. Remember to update <size> attribute as we
        push transitions.
        """
        self.buffer.state[self._cycle] = transition.state
        self.buffer.action[self._cycle] = transition.action
        self.buffer.reward[self._cycle] = transition.reward
        self.buffer.next_state[self._cycle] = transition.next_state
        self.buffer.terminal[self._cycle] = transition.terminal
        self._cycle = (self._cycle+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def sample(self, batchsize, *args):
        """ Uniformly sample a batch of transitions from the buffer. If
        batchsize is less than the number of valid transitions in the buffer
        return None. The return value must be a Transition object with batch
        of state, actions, .. etc.
            Return: T(states, actions, rewards, terminals, next_states)
        """
        if batchsize > self.size:
            return None

        idxs = np.random.randint(0, self.size, size = batchsize)
        batch = (self.buffer.state[idxs],
                    self.buffer.action[idxs],
                    self.buffer.reward[idxs],
                    self.buffer.next_state[idxs],
                    self.buffer.terminal[idxs])

        return self.Transition(*batch)

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError
