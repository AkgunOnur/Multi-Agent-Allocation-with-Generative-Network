import os
import numpy as np
import torch
from torch.optim import Adam

from .base import BaseAgent
from sac_discrete.model import TwinnedQNetwork, CateoricalPolicy
from sac_discrete.utils import disable_gradients


class SacdAgent(BaseAgent):

    def __init__(self, env, num_steps=100000, batch_size=128,
                 lr=0.0003, memory_size=150000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=200,
                 update_interval=4, target_update_interval=10,
                 use_per=True, dueling_net=True, num_eval_steps=125000,
                 max_episode_steps=100000, log_interval=10, eval_interval=500, max_iteration_steps=250,
                 device='cpu', seed=0):
        super().__init__(
            env, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps, max_iteration_steps,
            log_interval, eval_interval, device, seed)
        N_action = self.env.n_action ** 2
        np.set_printoptions(precision=2)
        # Define networks.
        self.policy = CateoricalPolicy(
            self.env.N_frame * (self.env.n_agents + 1) + 1, N_action).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.N_frame * (self.env.n_agents + 1) + 1, N_action,
            dueling_net=dueling_net).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            self.env.N_frame * (self.env.n_agents + 1) + 1, N_action,
            dueling_net=dueling_net).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = \
            -np.log(1.0 / N_action) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, agent_obs, device):
        # Act with randomness.
        with torch.no_grad():
            action, _, _ = self.policy.sample(agent_obs, device)

        return action.item()

    def exploit(self, agent_obs, device):
        # Act without randomness.
        with torch.no_grad():
            action = self.policy.act(agent_obs, device)
        return action.item()

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.long())
        curr_q2 = curr_q2.gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(
                next_states, self.device)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        # print ("batch: ", batch)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(
            states, self.device)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir, episode_number):
        super().save_models(save_dir, episode_number)
        self.policy.save(os.path.join(
            save_dir, 'policy_' + str(episode_number) + '.pth'))
        self.online_critic.save(os.path.join(
            save_dir, 'online_critic_' + str(episode_number) + '.pth'))
        self.target_critic.save(os.path.join(
            save_dir, 'target_critic_' + str(episode_number) + '.pth'))