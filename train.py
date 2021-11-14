import sys
import os
import torch
import time
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from graph import *

from point_mass_env import AgentFormation
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from utils import *


def generate_maps(seed = 7, N_SAMPLES=250):
    np.random.seed(seed)
    N_SAMPLES = N_SAMPLES
    gen_map_list = []
    for i in range(N_SAMPLES):
        p1 = np.random.uniform(0.65, 0.8)
        p2 = np.random.uniform(0.025, 0.1)
        gen_map = np.random.choice(3, (10,10), p=[p1, 1-p1-p2, p2])
        gen_map_list.append(gen_map)

    return gen_map_list

def main():

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_steps', default=5000, type=int, help='number of test iterations')
    parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--train_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='seed number for test')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--n_samples', default=250, type=int, help='seed number for test')
    args = parser.parse_args()

    total_procs = 0
    model_dir = "saved_models"
    new_graph = {}

    gen_map_list = generate_maps(seed=args.seed, N_SAMPLES=args.n_samples)

    for index1 in range(args.n_samples):
        gen_map = gen_map_list[index1]
        N_prize = np.sum(gen_map==2)
        total_procs += args.n_procs
        train_env = SubprocVecEnv([make_env(total_procs, gen_map) for j in range(args.n_procs)])

        # train_env = VecMonitor(train_env, filename = model_dir)
        # eval_env = SubprocVecEnv([make_env(total_procs, gen_map, max_steps=1000) for j in range(args.n_procs)])
        # eval_env = VecMonitor(eval_env, filename = model_dir)
        eval_env = AgentFormation(generated_map=gen_map, max_steps=1000)
        model = A2C('MlpPolicy', train_env, verbose=0)
        for experiment in range(args.train_episodes):
            # it is recommended to run several experiments due to variability in results
            train_env.reset()
            start = time.time()
            model.learn(total_timesteps=args.train_steps)

        train_env.close()
        mean_reward1, _  = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
        # mean_reward1 = 10.0
        elapsed_time = time.time() - start
        print ("\nT/Index: {0}, Mean Reward: {1:.4}, N.Prize: {2} Elapsed time: {3:.5}" \
            .format(index1, mean_reward1, N_prize, elapsed_time))
        
        if mean_reward1 < -40.0:
            continue
        else:
            new_graph[index1] = {}
            with open('nodes_to_train.pickle', 'wb') as handle:
                pickle.dump(new_graph, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    #     for j, index2 in enumerate(sorted_indices[::-1]):
        for index2 in range(args.n_samples):
            if index1 != index2:
                gen_map2 = gen_map_list[index2]
                N_prize2 = np.sum(gen_map2==2)
                start = time.time()
                # eval_env = SubprocVecEnv([make_env(total_procs, gen_map2, max_steps=1000) for j in range(args.n_procs)])
                # eval_env = VecMonitor(eval_env, filename = model_dir)
                eval_env = AgentFormation(generated_map=gen_map2, max_steps=1000)
                mean_reward2, _  = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
                elapsed_time = time.time() - start
                print ("E/Index: {0}, Mean Reward: {1:.4}, N.Prize: {2} Elapsed time: {3:.5}" \
                .format(index2, mean_reward2, N_prize2, elapsed_time))
                if mean_reward1 > mean_reward2:
                    new_graph[index1][index2] = mean_reward1 - mean_reward2


if __name__ == "__main__":
    main()

