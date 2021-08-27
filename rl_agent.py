import os
import torch
import time
import argparse
import glob
import re
import pickle
import numpy as np
import csv

from dqn_model import *

from point_mass_formation import AgentFormation
from read_maps import fa_regenate


def parameters():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = './saved_models'
    load_model_dir = './models'

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    parser = argparse.ArgumentParser(description='RL trainer')

    parser.add_argument('--visualization', default=False, type=bool, help='number of training episodes')
    # test
    parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    parser.add_argument('--load_model', default=load_model_dir, help='number of training episodes')
    parser.add_argument('--test_iteration', default=25, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--test_model_no', default=0, help='single model to evaluate')
    parser.add_argument('--test_model_level', default="easy", help='single model level to evaluate')
    # training
    parser.add_argument('--num_episodes', default=1000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='number of steps to update the policy')
    parser.add_argument('--eval_interval', type=int, default=50, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=25, help='After how many steps to start training')
    # model
    parser.add_argument('--resume', default=False, action='store_true', help='to continue the training')
    parser.add_argument('--model_dir', default=model_dir, help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='Batch size to train')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=20, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    parser.add_argument('--n_states', type=int, default=7350, help='Number of states after convolution layer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train')
    parser.add_argument('--memory_size', type=int, default=100000, help='Buffer memory size')
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    parser.add_argument('--out_shape', type=int, default=20, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=100, help='Hidden size dimension')
    parser.add_argument('--device', default=device, help='device')

    return parser.parse_args()

class rl:
    def __init__(self, current_scale):
        self.args = parameters()
        self.current_scale = current_scale
        self.iteration = 0
        self.best_reward = -np.inf
        
        # Create environments.
        self.env = AgentFormation(visualization=self.args.visualization)

        #create RL agent
        self.dqn = DQN(self.args)

    def train(self, coded_fake_map, current_scale, iteration):
        self.iteration = iteration
        ds_map, obstacle_map, prize_map, agent_obs = fa_regenate(coded_fake_map)

        #reset environment
        self.env.reset(ds_map, obstacle_map, prize_map, agent_obs)

        #get action
        action = self.dqn.choose_action(agent_obs) # output is between 0 and 7
        n_agents = action + 1 # number of allowable agents is 1 to 8

        episode_reward, done, agent_next_obs = self.env.step(n_agents)
        
        if self.args.visualization:
            self.env.close()
        
        self.dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)

        if  self.iteration > self.args.start_step and self.iteration % self.args.update_interval == 0:
            self.dqn.learn()
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.dqn.save_models(os.path.join(self.args.model_dir, 'train'), current_scale, 1)

        #print(f'Train Scale- {current_scale} | Iteration: {self.iteration} | Episode Reward: {round(episode_reward, 2)}')