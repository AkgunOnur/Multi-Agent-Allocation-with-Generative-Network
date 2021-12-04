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


def generate_maps(N_maps=1):
    gen_map_list = []
    
    p1 = 0.6  #np.random.uniform(0.65, 0.8)
    p2 = 0.05 #np.random.uniform(0.025, 0.1)
    for i in range(N_maps):
        gen_map = np.random.choice(3, (10,10), p=[p1, 1-p1-p2, p2])
        gen_map_list.append(gen_map)

    return gen_map_list


def curriculum_design(gen_map_list, rng, level = "easy"):
    modified_map_list = []
    coeff = 1.0
    if level == "easy":
        coeff = 0.1
    elif level == "medium":
        coeff = 0.5

    for gen_map in gen_map_list:
        obstacles = np.argwhere(gen_map == 1)
        rewards = np.argwhere(gen_map == 2)
        modified_map = np.copy(gen_map)

        n_samples = len(obstacles) - int(len(obstacles) * coeff)
        obstacle_to_remove = rng.choice(obstacles, size=(n_samples,), replace=False)
        for obs_loc in obstacle_to_remove:
            modified_map[obs_loc[0], obs_loc[1]] = 0
        modified_map_list.append(modified_map)
        
    return modified_map_list


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
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
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
        

def main(args):
    np.random.seed(args.seed)
    rng = default_rng(args.seed)

    model_dir = args.out + "/saved_models"
    curriculum_list = ["easy", "medium", "target"]
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(args.n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)
    model_name = "PPO"
    N_eval = 2000
    model_def = PPO

    for map_ind in range(args.n_maps):
        print ("Current map index: ", map_ind)
        gen_map = generate_maps()
        easy_map = curriculum_design(gen_map, rng, level = "easy")
        medium_map = curriculum_design(gen_map, rng, level = "medium")

        if args.nocurriculum:
            print ("No Curriculum")
            level = "target"
            train_env = make_vec_env(lambda: AgentFormation(generated_map=gen_map), n_envs=args.n_procs, monitor_dir=args.out + "/model_outputs_" + level + str(map_ind), vec_env_cls=SubprocVecEnv)
            train_env.reset()
            model = model_def('CnnPolicy', train_env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            eval_env = AgentFormation(generated_map=gen_map, max_steps=1000)
            callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval // args.n_procs + 1, log_path  = args.out + "/" + model_name + "_" + level + "_log",
                                    best_model_save_path = model_dir + "/best_model_" + level, verbose=1)

            start = time.time()
            nocurriculum_train_steps = int(3*args.train_steps)
            model.learn(total_timesteps=nocurriculum_train_steps, tb_log_name=model_name + "_run_" + level)
            elapsed_time = time.time() - start
            print (f"Elapsed time: {elapsed_time:.5}")

        else:
            for level in curriculum_list:
                print (f"\nCurriculum Level: {level}")
                if level == "easy":
                    current_map = np.copy(easy_map)
                elif level == "medium":
                    current_map = np.copy(medium_map)
                else:
                    current_map = np.copy(gen_map)

                
                train_env = make_vec_env(lambda: AgentFormation(generated_map=current_map), n_envs=args.n_procs, monitor_dir=args.out + "/model_outputs_" + level + str(map_ind), vec_env_cls=SubprocVecEnv)
                # train_env = AgentFormation(generated_map=current_map)
                # train_env = SubprocVecEnv([make_env(current_map) for i in range(args.n_procs)])
                train_env.reset()
                model = model_def('CnnPolicy', train_env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
                # model = model_def('CnnPolicy', train_env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log="./" + model_name + "_tensorboard/")
                eval_env = AgentFormation(generated_map=current_map, max_steps=1000)
                callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval // args.n_procs + 1, log_path  = args.out + "/" + model_name + "_" + level + "_log",
                                        best_model_save_path = model_dir + "/best_model_" + level, verbose=1)

                if level == "medium":
                    model.load(model_dir + "/best_model_easy/best_model.zip", verbose=1)
                    model.set_env(train_env)
                elif level == "target":
                    model.load(model_dir + "/best_model_medium/best_model.zip", verbose=1)
                    model.set_env(train_env)
                
                start = time.time()
                model.learn(total_timesteps=args.train_steps, tb_log_name=model_name + "_run_" + level, callback=callback)
                # x, y = ts2xy(load_results(model_dir), 'timesteps')
                # train_reward = np.mean(y)
                # eval_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
                # mean_reward1 = np.random.uniform(-80, 20)
                elapsed_time = time.time() - start
                print (f"Elapsed time: {elapsed_time:.5}")
            

        train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_steps', default=20000, type=int, help='number of test iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    # parser.add_argument('--train_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=12, type=int, help='number of processes to execute')
    parser.add_argument('--n_maps', default=10, type=int, help='number of maps to train')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--out', default="output", type=str, help='the output folder')
    parser.add_argument('--nocurriculum', default = False, action='store_true', help='train on only the target map')

    args = parser.parse_args()
    args.out = "output_cur"
    main(args)

    args.nocurriculum = True
    args.out = "output_nocur"
    main(args)

