import torch
import numpy as np
from copy import deepcopy

from dqn.replaybuffer.uniform import UniformBuffer


class BaseDQN(torch.nn.Module):
    """ Base Class for DQN and agents.
        Arguments:
            - valunet: Neural network to estimate values
            - nact: Number of actions (and outputs)
    """

    Transition = UniformBuffer.Transition

    def __init__(self, valuenet, nact):
        super().__init__()
        self.valuenet = valuenet
        self.nact = nact
        self.targetnet = deepcopy(valuenet)

    def greedy_policy(self, state):
        """ Return the action that has the highest value for the given state
        """
        values = self.valuenet(state)
        action = torch.argmax(values)

        return action.item()

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

    def e_greedy_policy(self, state, epsilon):
        """ Randomly (Bernoulli distribution with p equals to epsilon) select a
        random action (ranging from 0 to nact) or select the greedy action.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.nact)
        else:
            return self.greedy_policy(state)

    def push_transition(self, transition):
        raise NotImplementedError

    def loss(self, batch, gamma):
        raise NotImplementedError

    def update_target(self):
        """ Update the target network by setting its parameters to valuenet
        parameters """
        self.targetnet.load_state_dict(self.valuenet.state_dict())

    def evaluate(self, eval_episode, env, device):
        """ Agent evaluation function. Evaluate the current greedy policy for
        args.eval_episode many "full" episodes. Return the mean episodic reward
        (average of rewards per episode).
        """
        self.eval()
        eval_reward_list = []

        for eps in range(eval_episode):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                torch_state = self.state_to_torch(state, device)
                action = self.greedy_policy(torch_state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward 
                state = next_state

            eval_reward_list.append(episode_reward)

        #  /$$$$$$$$ /$$$$$$ /$$       /$$
        # | $$_____/|_  $$_/| $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$$$$     | $$  | $$      | $$
        # | $$__/     | $$  | $$      | $$
        # | $$        | $$  | $$      | $$
        # | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
        # |__/      |______/|________/|________/
        #raise NotImplementedError

        return np.mean(eval_reward_list).item()

    @staticmethod
    def batch_to_torch(batch, device):
        """ Convert numpy transition into a torch transition
        Note: Dtype of actions is "long" while the remaining dtypes are
        "float32" """
        return BaseDQN.Transition(
            *(torch.from_numpy(x).type(dtype).to(device)
              for x, dtype in zip(
                batch,
                (torch.float,
                 torch.long,
                 torch.float32,
                 torch.float32,
                 torch.float32)))
        )

    @staticmethod
    def state_to_torch(state, device):
        """ Convert numpy state into torch state
        """
        return torch.from_numpy(state).float().to(device)
