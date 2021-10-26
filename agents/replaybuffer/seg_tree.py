""" Sum Tree implementation for the prioritized
replay buffer.
"""

import numpy as np
import math


class SegTree():
    """ Base Segment Tree Class with binary heap implementation that push
    values as a Queue(FIFO).
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._cycle = 0
        self.size = 0

        self.index = 0
        self.tree_level = int(np.ceil(np.log2(capacity)) + 1)
        self.tree = np.zeros(2 * self.capacity - 1)
        #start = self.tree.size // 2
        #end = start + self.capacity

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def push(self, value):
        """ Push a value into the tree by calling the update method. Push
        function overrides values when the tree is full """

        self.update(self.index, value)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def update(self, value):
        raise NotImplementedError


class SumTree(SegTree):
    """ A Binary tree with the property that a parent node is the sum of its
    two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)

    def get(self, value):
        """ Return the index (ranging from 0 to max capcaity) that corresponds
        to the given value """
        if value > self.tree[0]:
            raise ValueError("Value is greater than the root")

        indx = 0

        while True:
            left = 2 * indx + 1
            right = left + 1

            if left >= len(self.tree):
                leaf_index = indx
                break

            if value <= self.tree[left]:
                indx = left
            else:
                value = value - self.tree[left]
                indx = right

        val_index = leaf_index - self.capacity + 1

        return val_index


        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capacity) with the given value
        """
        assert value >= 0, "Value cannot be negative"

        index += self.tree.size // 2

        while True:
            self.tree[index] = value
            if index == 0:
                break
            right = ((index + 1) // 2) * 2
            left = right - 1
            value = self.tree[left] + self.tree[right]
            index = (index - 1) // 2

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError


class MinTree(SegTree):
    """ A Binary tree with the property that a parent node is the minimum of
    its two children.
        Arguments:
            - capacity: Maximum size of the tree. It also the number of leaf
            nodes in the tree.
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree[:] = np.inf

    def update(self, index, value):
        """ Update the value of the given index (ranging from 0 to max
        capcaity) with the given value
        """
        assert value >= 0, "Value cannot be negative"

        index += self.tree.size // 2

        while True:
            self.tree[index] = value
            if index == 0:
                break
            right = ((index + 1) // 2) * 2
            left = right - 1
            value = min(self.tree[left], self.tree[right])
            index = (index - 1) // 2

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    @property
    def minimum(self):
        """ Return the minimum value of the tree (root node). Complexity: O(1)
        """
        return self.tree[0]
        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError
