import torch
import numpy as np
from copy import deepcopy
from collections import namedtuple
import os

from .model import DQN
from dqn.common import linear_annealing, exponential_annealing, PrintWriter


class Trainer:
	""" Training class that organize evaluation, update, and transition
	gathering.
		Arguments:
			- args: Parser arguments
			- agent: RL agent object
			- opt: Optimizer that optimizes agent's parameters
			- env: Gym environment
	"""

	def __init__(self, args, agent, opt, env):

		self.env = deepcopy(env)
		self.eval_env = deepcopy(env)
		self.args = args
		self.agent = agent
		self.opt = opt

		self.train_rewards = []
		self.eval_rewards = []
		self.td_loss = []
		self._writer = PrintWriter(flush=True)

		self.checkpoint_reward = -np.inf
		self.agent.to(args.device)
		self.epsilon = self.args.epsilon_init


		self.linear_annealing = linear_annealing
		self.exponential_annealing = exponential_annealing
		self.epsilon_decay  = args.epsilon_decay
		self.epsilon_range = args.epsilon_rang
		
		#  /$$$$$$$$ /$$$$$$ /$$       /$$
		# | $$_____/|_  $$_/| $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$$$$     | $$  | $$      | $$
		# | $$__/     | $$  | $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
		# |__/      |______/|________/|________/
		#raise NotImplementedError

	def __call__(self):
		""" Start training """
		for ix, trans in enumerate(self):
			self.evaluation(ix)
			self.agent.push_transition(trans)
			self.update(ix)
			self.writer(ix)

	def evaluation(self, ix):
		""" Evaluate the agent if the index "ix" is at the evaluation period. 
		If "save_model" is given the current best model is saved based on the
		evaluation score. Evaluation score appended into the "eval_rewards"
		list to keep track of evaluation scores.
			Arguments:
				- ix: Training iteration
			Raise:
				- FileNotFoundError: If "save_model" is given in arguments and
				directory given by "model_dir" does not exist
		"""
		if ix % self.args.eval_period == 0:

			self.eval_rewards.append(
				self.agent.evaluate(self.args.eval_episode,
									self.eval_env,
									self.args.device))
			if self.eval_rewards[-1] > self.checkpoint_reward and self.args.save_model:
				self.checkpoint_reward = self.eval_rewards[-1]
				model_id = "{}_{:6d}_{:6.3f}.b".format(
					self.agent.__class__.__name__,
					ix,
					self.eval_rewards[-1]).replace(" ", "0")
				if not os.path.exists(self.args.model_dir):
					raise FileNotFoundError(
						"No directory as {}".format(self.args.model_dir))
				torch.save(dict(
					model=self.agent.state_dict(),
					optim=self.opt.state_dict(),
				), os.path.join(self.args.model_dir, model_id)
				)

	def update(self, ix):
		""" One step updating function. Update the agent in training mode,
		clip gradient if "clip_grad" is given in args, and keep track of td
		loss. Check for the training index "ix" to start the update.
			Arguments:
				- ix: Training iteration
		"""
		self.agent.train()
		if ix > self.args.start_update:
			pass

			if ix % self.args.target_update_period == 0:
				self.agent.targetnet.load_state_dict(self.agent.valuenet.state_dict())
		
			self.opt.zero_grad()
			td_error = self.agent.loss(self.args.batch_size, self.args.gamma)
			self.td_loss.append(td_error.item())
			td_error.backward()

			if self.args.clip_grad:
				for param in self.agent.valuenet.parameters():
					param.grad.data.clamp_(-1, 1)

			self.opt.step()

			return td_error.item()

		
		#  /$$$$$$$$ /$$$$$$ /$$       /$$
		# | $$_____/|_  $$_/| $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$$$$     | $$  | $$      | $$
		# | $$__/     | $$  | $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
		# |__/      |______/|________/|________/
		#raise NotImplementedError

	def writer(self, ix):
		""" Simple writer function that feeds PrintWriter with statistics """
		if ix % self.args.write_period == 0:
			self._writer(
				{
					"Iteration: {:7}": ix,
					"Train reward: {:7.3f}": np.mean(self.train_rewards[-20:]),
					"Eval reward: {:7.3f}": self.eval_rewards[-1],
					"TD loss: {:7.3f}": np.mean(self.td_loss[-100:]),
					"Epsiode: {:4}": len(self.train_rewards)
				},
			)

	def __iter__(self):
		""" Experience source function that yields a transition at every
		iteration """
		
		Transition = namedtuple("Transition", "state action reward next_state terminal")
		state = self.env.reset()
		episode_reward = 0

		for i in range(self.args.n_iterations):

			torch_state = self.agent.state_to_torch(state, self.args.device)
			action = self.agent.e_greedy_policy(torch_state, self.epsilon)
			next_state, reward, done, _ = self.env.step(action)
			episode_reward += reward
			yield Transition(state, action, reward, next_state, done)
			state = next_state

			if done:
				state = self.env.reset()
				self.train_rewards.append(episode_reward)
				episode_reward = 0

				if self.epsilon_decay is not None:
					self.epsilon = next(self.exponential_annealing(self.epsilon,
											self.args.epsilon_min, 
											self.epsilon_decay))	

				if self.epsilon_range is not None:
					self.epsilon = next(self.linear_annealing(self.epsilon,
										self.args.epsilon_min, 
										self.epsilon_range))

		#  /$$$$$$$$ /$$$$$$ /$$       /$$
		# | $$_____/|_  $$_/| $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$$$$     | $$  | $$      | $$
		# | $$__/     | $$  | $$      | $$
		# | $$        | $$  | $$      | $$
		# | $$       /$$$$$$| $$$$$$$$| $$$$$$$$
		# |__/      |______/|________/|________/
		#raise NotImplementedError
