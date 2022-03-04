from fileinput import filename
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




def generate_maps(N_maps=1, map_lim=10):
    gen_map_list = []
    
    p1 = 0.72  #np.random.uniform(0.65, 0.8)
    p2 = 0.03 #np.random.uniform(0.025, 0.1)
    for i in range(N_maps):
        gen_map = np.random.choice(3, (map_lim,map_lim), p=[p1, 1-p1-p2, p2])
        gen_map_list.append(gen_map)

    return gen_map_list


def curriculum_design(gen_map_list, rng, coeff=1.0):
    modified_map_list = []

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


def init_network(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
        
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
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.size()) == 3:
            observations = torch.reshape(observations, (1, *observations.size()))
        return self.linear(self.cnn(observations))



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

    for map_ind in range(args.n_maps):
        print ("\nCurrent map index: ", map_ind)

        if args.nocurriculum:
            map_list = []
            print ("No Curriculum")
            level = curriculum_list[-1]
            map_list.append(generated_map_list[level][map_ind])
            
            # target_map = generated_map_list[2][map_ind]
            N_reward = len(np.argwhere(map_list[0] == 2))
            max_possible_reward = N_reward - agent_step_cost * 100
            print ("max_possible_reward: ", max_possible_reward)
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_possible_reward, verbose=1)

            current_map_size = args.map_lim
            target_map_lim = args.map_lim

            train_folder = args.out + "/train_" + level + "_map_" + str(map_ind)
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)

            
            val_folder = args.out + "/val_" + level + "_map_" + str(map_ind)
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)

            # train_env = DummyVecEnv([lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize)])
            train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=args.map_lim, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*args.map_lim), n_envs=n_process, monitor_dir=train_folder, vec_env_cls=SubprocVecEnv)
            # train_env = AgentFormation(generated_map=gen_map, map_lim=args.map_lim, max_steps=250)
            # train_env = Monitor(train_env, train_folder + "/monitor.csv")
            # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
            train_env.reset()
            # model = model_def('MlpPolicy', train_env, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
            model = model_def('MlpPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                            n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            
            # policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=net_list[0], features_extractor_kwargs=dict(features_dim=net_list[0][0]), activation_fn=activation_list[0])
            # model = model_def('CnnPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                        # n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            
            eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=val_folder, vec_env_cls=SubprocVecEnv)
            # eval_env = AgentFormation(generated_map=target_map, map_lim=args.map_lim, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*args.map_lim)
            # eval_env = VecNormalize(eval_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)

            callback = EvalCallback(eval_env=eval_env, callback_on_new_best=stop_callback,  eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + "_map_" + str(map_ind) +"_log",
                                    best_model_save_path = model_dir + "/best_model_" + level + "_map_" + str(map_ind), deterministic=False, verbose=1)

            start = time.time()
            # nocurriculum_train_steps = int(len(curriculum_list)*args.train_timesteps)
            model.learn(total_timesteps=args.train_timesteps, tb_log_name=model_name + "_run_" + level, callback=callback)
            model.save(model_dir + "/last_model_" + level + "_map_" + str(map_ind))
            elapsed_time = time.time() - start
            print (f"Elapsed time: {elapsed_time:.5}")

        else:
            env_list = []
            eval_list = []
            max_reward_list = []
            for index, level in enumerate(curriculum_list):
                map_list = [] # each map is given to the environment as a list containing a single map
                # current_map_size = curiculum_map_sizes[index]
                current_map_size = args.map_lim
                map_list.append(generated_map_list[level][map_ind])
                N_reward = len(np.argwhere(map_list[0] == 2))
                max_possible_reward = N_reward - agent_step_cost * 100
                max_reward_list.append(max_possible_reward)

                train_folder = args.out + "/train_" + level + "_map_" + str(map_ind)
                if not os.path.exists(train_folder):
                    os.makedirs(train_folder)

                val_folder = args.out + "/val_" + level + "_map_" + str(map_ind)
                if not os.path.exists(val_folder):
                    os.makedirs(val_folder)
                
                # train_env = DummyVecEnv([lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize)])
                # train_env = Monitor(train_env, train_folder + "/monitor.csv")
                # train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False, clip_obs=1.)
                train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize), n_envs=n_process, monitor_dir=train_folder, vec_env_cls=SubprocVecEnv)
                env_list.append(train_env)

                eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=val_folder, vec_env_cls=SubprocVecEnv)
                # eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False, clip_obs=1.)
                eval_list.append(eval_env)

            
            for index, level in enumerate(curriculum_list):
                print (f"\nCurriculum Level: {level}")
                print ("max_possible_reward: ", max_reward_list[index])
                stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_reward_list[index], verbose=1)
                env_list[index].reset()
                previous_index = np.clip(index-1, 0, len(curriculum_list) - 1)

                model_name = "best_model_" + curriculum_list[previous_index] + str(map_ind)
                if os.path.exists(model_dir + "/" +  model_name + "/best_model.zip"):
                    model = model_def.load(model_dir + "/" +  model_name + "/best_model", verbose=1)
                else:
                    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
                    model = model_def('MlpPolicy', env_list[index], n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                                        n_steps = ns_list[0],  policy_kwargs=policy_kwargs)

                # policy_kwargs = dict(features_extractor_class=CNN_Network,activation_fn=activation_list[0], net_arch=net_list[0], 
                                    # features_extractor_kwargs=dict(features_dim=128))
                # model = model_def('CnnPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                #             n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")

                model.set_env(env_list[index])
                
                callback = EvalCallback(eval_env=eval_list[index], callback_on_new_best=stop_callback, eval_freq = N_eval // n_process,
                                        best_model_save_path = model_dir + "/best_model_" + level + "_map_" + str(map_ind), deterministic=False, verbose=1)

                # if current level is target, do not appy stop callback
                # if level == curriculum_list[-1]: 
                #     callback = EvalCallback(eval_env=eval_list[index], eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                #                         best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=1)
                # else:
                #     callback = EvalCallback(eval_env=eval_list[index], callback_on_new_best=stop_callback, eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                #                         best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=1)

                
                
                start = time.time()
                model.learn(total_timesteps=args.train_timesteps, callback=callback)
                model.save(model_dir + "/last_model_" + level + "_map_" + str(map_ind))
                # stats_path = os.path.join(model_dir, "vec_normalize.pkl")
                # env_list[index].save(stats_path)
                # eval_reward, _  = evaluate_policy(model, eval_list[index], n_eval_episodes=args.eval_episodes)
                elapsed_time = time.time() - start
                print (f"Elapsed time: {elapsed_time:.5}")
            

        train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_timesteps', default=200000, type=int, help='number of train iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--map_lim', default=20, type=int, help='width and height of the map')
    parser.add_argument('--n_procs', default=8, type=int, help='number of processes to execute')
    parser.add_argument('--n_maps', default=5, type=int, help='number of maps to train')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--out', default="output_nocur_20_200k", type=str, help='the output folder')
    parser.add_argument('--map_file', default="saved_maps_20", type=str, help='the output folder')
    parser.add_argument('--nocurriculum', default = True, action='store_true', help='train on only the target map')
    parser.add_argument('--visualize', default = False, action='store_true', help='to visualize')
    args = parser.parse_args()
    
    main(args)

