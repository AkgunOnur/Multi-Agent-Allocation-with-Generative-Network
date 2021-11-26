import sys
import os
import torch
import gym
import time
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from graph import *

from point_mass_env import AgentFormation
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *


def generate_maps(seed = 7):
    np.random.seed(seed)
    p1 = 0.7  #np.random.uniform(0.65, 0.8)
    p2 = 0.05 #np.random.uniform(0.025, 0.1)
    gen_map = np.random.choice(3, (10,10), p=[p1, 1-p1-p2, p2])

    return gen_map


def curriculum_design(gen_map, level = "easy", seed=7):
    coeff = 1.0
    if level == "easy":
        coeff = 0.1
    elif level == "medium":
        coeff = 0.5

    obstacles = np.argwhere(gen_map == 1)
    rewards = np.argwhere(gen_map == 2)
    modified_map = np.copy(gen_map)

    rng = default_rng(seed)
    n_samples = len(obstacles) - int(len(obstacles) * coeff)
    obstacle_to_remove = rng.choice(obstacles, size=(n_samples,), replace=False)
    for obs_loc in obstacle_to_remove:
        modified_map[obs_loc[0], obs_loc[1]] = 0
    
    return modified_map


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),)
        

def main():

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_steps', default=100000, type=int, help='number of test iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    # parser.add_argument('--train_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='seed number for test')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    # parser.add_argument('--n_samples', default=250, type=int, help='seed number for test')
    args = parser.parse_args()

    model_dir = "saved_models"

    gen_map = generate_maps(seed=args.seed)

    easy_map = curriculum_design(gen_map, level = "easy")
    medium_map = curriculum_design(gen_map, level = "medium")
    curriculum_list = ["easy", "medium", "target"]
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(args.n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)
    model_name = "A2C"
    N_eval = 2000
    
    for level in curriculum_list:
        if level == "easy":
            current_map = np.copy(easy_map)
        elif level == "medium":
            current_map = np.copy(medium_map)
        else:
            current_map = np.copy(gen_map)

        
        train_env = make_vec_env(lambda: AgentFormation(generated_map=current_map), n_envs=args.n_procs, monitor_dir="model_outputs_" + level, vec_env_cls=SubprocVecEnv)
        # train_env = AgentFormation(generated_map=current_map)
        # train_env = SubprocVecEnv([make_env(current_map) for i in range(args.n_procs)])
        train_env.reset()
        model = A2C('MlpPolicy', train_env, verbose=0, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + model_name + "_tensorboard/")
        # model = A2C('CnnPolicy', train_env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log="./" + model_name + "_tensorboard/")
        eval_env = AgentFormation(generated_map=current_map, max_steps=1000)
        callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval // args.n_procs + 1, log_path  = model_name + "_" + level + "_log",
                                best_model_save_path = model_dir + "/best_model_" + level, verbose=1)

        if level == "medium":
            model.load(model_dir + "/best_model_easy/best_model.zip", verbose=1)
        elif level == "target":
            model.load(model_dir + "/best_model_medium/best_model.zip", verbose=1)
        
        start = time.time()
        model.learn(total_timesteps=args.train_steps, tb_log_name=model_name + "_run_" + level, callback = callback)
        # x, y = ts2xy(load_results(model_dir), 'timesteps')
        # train_reward = np.mean(y)
        # eval_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
        # mean_reward1 = np.random.uniform(-80, 20)
        elapsed_time = time.time() - start
        print ("\nT/Exp: {0}, Elapsed time: {1:.5}".format(level, elapsed_time))

    train_env.close()

if __name__ == "__main__":
    main()

