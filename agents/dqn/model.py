import torch
import numpy as np

from dqn.replaybuffer.uniform import UniformBuffer
from dqn.base_dqn import BaseDQN


class DQN(BaseDQN):
    """ Deep Q Network that uses the target network and uniform replay buffer.
        Arguments:
            - valunet: Neural network to estimate values
            - nact: Number of actions (and outputs)
            - buffer_args: Remaning positional arguments to feed replay buffer
    """

    def __init__(self, valuenet, nact, *buffer_args):
        super().__init__(valuenet, nact)
        self.buffer = UniformBuffer(*buffer_args)
        self.device = torch.device("cpu")

    def push_transition(self, transition):
        self.buffer.push(transition)

    def loss(self, batch, gamma) -> torch.Tensor:
        """ DQN loss that uses the target network to estimate target
        values.
            Arguments:
                - batch: Batch of transition as Transition namedtuple defined
                in BaseDQN class
                - gamma: Discount factor
            Return:
                td_error tensor: MSE loss (L1, L2 or smooth L1) of the target
                and predicted values
        """
        batch = self.buffer.sample(batch)
        batch = self.batch_to_torch(batch, self.device)

        with torch.no_grad():
            next_values = self.targetnet(batch.next_state)
            next_values = torch.max(next_values, dim=1, keepdim=True)[0]

        current_values = self.valuenet(batch.state)
        current_values = current_values.gather(1, batch.action)

        target_value = next_values*(1 - batch.terminal)*gamma + batch.reward
        td_error = torch.nn.functional.smooth_l1_loss(current_values, target_value)

        return td_error

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError
