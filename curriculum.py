from fileinput import filename
import sys
import os
import torch
import gym
import time
import pickle
import argparse
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from numpy.random import default_rng
from graph import *

from point_mass_env import AgentFormation
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
# from monitor_new import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from utils import *



def init_network(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class CNN_Network(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNN_Network, self).__init__(observation_space, features_dim)
        init_ = lambda m: init_network(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu')) 
        n_input_channels = observation_space.shape[0]

        self.feat_extract = nn.Sequential(
                init_(nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(), nn.Flatten()
                )
        
        with torch.no_grad():
            n_flatten = self.feat_extract(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            init_(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
            init_(nn.Linear(features_dim, features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.size()) == 3:
            observations = torch.reshape(observations, (1, *observations.size()))

        cnn_out = self.feat_extract(observations)
        lin_out = self.linear(cnn_out)
        return lin_out



def main(args):
    np.random.seed(args.seed)
    rng = default_rng(args.seed)
    agent_step_cost = 0.01

    
    # curriculum_list = ["easy", "medium", "target"]
    curriculum_list = ["level1","level2", "level3", "level4", "level5"]
    # curiculum_map_sizes = [10, 20, 30, 40]
    
    model_name = "PPO"
    model_def = PPO
    N_eval = 1000

    activation_list = [nn.Tanh]
    gamma_list = [0.9]
    bs_list = [64]
    lr_list = [3e-4]
    net_list = [[64, 64]]
    ns_list = [2048]
    ne_list = [10]

    with open(args.map_file + '.pickle', 'rb') as handle:
        generated_map_list = pickle.load(handle)
    


    model_dir = args.out + "/saved_models"

    n_process = args.n_procs
    target_map_lim = args.map_lim

    for map_ind in range(1, args.n_maps):
        print ("\nCurrent map index: ", map_ind)
        # env_list = []
        # eval_list = []
        # max_reward_list = []
        training_iter = 0
        keep_training = True
        reward_list = defaultdict()
        level_list = []
        level = "level1"

        for index in range(len(curriculum_list)):
            reward_list[curriculum_list[index]] = -100.0        

        while (keep_training):
            print ("\ncurrent level: ", level)
            level_list.append(level)
            level_index = curriculum_list.index(level)
            map_list = [] # each map is given to the environment as a list containing a single map
            # current_map_size = curiculum_map_sizes[index]
            current_map_size = args.map_lim
            map_list.append(generated_map_list[level][map_ind])
            N_reward = len(np.argwhere(map_list[0] == 2))
            max_possible_reward = N_reward - agent_step_cost * 100
            # max_reward_list.append(max_possible_reward)
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_possible_reward, verbose=1)
            

            train_folder = args.out + "/train_" + str(training_iter) + "_" + level + "_map_" + str(map_ind)
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)

            val_folder = args.out + "/val_" + str(training_iter) + "_" + level + "_map_" + str(map_ind)
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)

            train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=train_folder, vec_env_cls=SubprocVecEnv)
            eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=val_folder, vec_env_cls=SubprocVecEnv)

            train_env.reset()
            eval_env.reset()

            policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
            model = model_def('MlpPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                                        n_steps = ns_list[0],  policy_kwargs=policy_kwargs)

            if training_iter > 0:
                model_name = "best_model_" + str(training_iter - 1) + "_" + level_list[training_iter - 1] + "_map_" + str(map_ind)
                if os.path.exists(model_dir + "/" +  model_name + "/best_model.zip"):
                    print ("Model Loaded!")
                    model = model_def.load(model_dir + "/" +  model_name + "/best_model", verbose=1)
                    model.set_env(train_env)

            # env_list[level_index].reset()
            # eval_list[level_index].reset()
            # model.set_env(env_list[level_index])

            callback = EvalCallback(eval_env=eval_env, callback_on_new_best=stop_callback, eval_freq = N_eval // n_process,
                                    best_model_save_path = model_dir + "/best_model_" + str(training_iter) + "_" + level + "_map_" + str(map_ind), deterministic=False, verbose=1)

            
            start = time.time()
            model.learn(total_timesteps=args.train_timesteps, callback=callback)
            model.save(model_dir + "/last_model_" + level + "_map_" + str(map_ind))
            # stats_path = os.path.join(model_dir, "vec_normalize.pkl")

            # model_name = "best_model_" + str(training_iter) + "_" + level + "_map_" + str(map_ind)
            # if os.path.exists(model_dir + "/" +  model_name + "/best_model.zip"):
            #     print ("Current Best Model Loaded!")
            #     model = model_def.load(model_dir + "/" +  model_name + "/best_model", verbose=1)

            # eval_env.reset()
            # eval_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=False)
            reward_list[level] = callback.best_mean_reward
            updated_level_list = sorted(reward_list, key=reward_list.get, reverse=True)
            # print ("reward_list: ", reward_list)
            # print ("updated_level_list: ", updated_level_list)
            new_level_index = updated_level_list.index(level) + 1
            if new_level_index == len(updated_level_list):
                new_level_index = np.random.randint(len(updated_level_list))
                
            level = updated_level_list[new_level_index]
            # print ("new level: ", level)
            elapsed_time = time.time() - start
            print (f"Elapsed time: {elapsed_time:.5}")
            training_iter += 1

            if reward_list[curriculum_list[-1]] >= (max_possible_reward - 0.5): # if target reward achieved for the target level, stop training
                keep_training = False
                print ("Training stopped! Goodbye!")

        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=20000, type=int, help='number of train iterations')
    parser.add_argument('--eval_episodes', default=10, type=int, help='number of test iterations')
    parser.add_argument('--map_lim', default=20, type=int, help='width and height of the map')
    parser.add_argument('--n_procs', default=8, type=int, help='number of processes to execute')
    parser.add_argument('--n_maps', default=5, type=int, help='number of maps to train')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--out', default="output_cur_yeni_20_10k", type=str, help='the output folder')
    parser.add_argument('--map_file', default="yeni_haritalar_20", type=str, help='the output folder')
    parser.add_argument('--visualize', default = False, action='store_true', help='to visualize')
    args = parser.parse_args()
    
    main(args)

