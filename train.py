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
from monitor_new import Monitor
from stable_baselines3.common.callbacks import EvalCallback
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
    
    p1 = 0.7  #np.random.uniform(0.65, 0.8)
    p2 = 0.05 #np.random.uniform(0.025, 0.1)
    for i in range(N_maps):
        gen_map = np.random.choice(3, (map_lim,map_lim), p=[p1, 1-p1-p2, p2])
        gen_map_list.append(gen_map)

    return gen_map_list


def curriculum_design(gen_map_list, rng, level = "easy", coeff=1.0):
    modified_map_list = []
    # coeff = 1.0
    # if level == "easy":
    #     coeff = 0.1
    # elif level == "medium":
    #     coeff = 0.5

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
        

def main(args):
    np.random.seed(args.seed)
    rng = default_rng(args.seed)

    model_dir = args.out + "/saved_models"
    curriculum_list = ["easy", "medium", "target"]
    
    # train_env = SubprocVecEnv([make_env(easy_map) for j in range(args.n_procs)])
    # train_env = AgentFormation(generated_map=easy_map)
    # train_env = VecMonitor(train_env, filename = model_dir)
    model_name = "PPO"
    model_def = PPO
    N_eval = 1000

    with open(args.map_file + '.pickle', 'rb') as handle:
        easy_list, medium_list, gen_list = pickle.load(handle)
    

    for map_ind in range(args.n_maps):
        print ("\nCurrent map index: ", map_ind)

        gen_map = gen_list[map_ind]
        easy_map = easy_list[map_ind]
        medium_map = medium_list[map_ind]

        if args.nocurriculum:
            print ("No Curriculum")
            level = "target"

            current_folder = args.out + "/model_outputs_" + level + str(map_ind)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            train_env = make_vec_env(lambda: AgentFormation(generated_map=gen_map, map_lim=args.map_lim), n_envs=args.n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
            # train_env = AgentFormation(generated_map=gen_map, map_lim=args.map_lim, max_steps=250)
            # train_env = Monitor(train_env, current_folder + "/monitor.csv")
            train_env.reset()
            model = model_def('MlpPolicy', train_env, verbose=0, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            eval_env = AgentFormation(generated_map=gen_map, map_lim=args.map_lim)
            callback = EvalCallback(eval_env=eval_env, eval_freq = N_eval // args.n_procs, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                                    best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=1)

            start = time.time()
            nocurriculum_train_steps = int(3*args.train_epochs)
            model.learn(total_timesteps=nocurriculum_train_steps, tb_log_name=model_name + "_run_" + level, callback=callback)
            model.save(model_dir + "/last_model_" + level + str(map_ind))
            elapsed_time = time.time() - start
            print (f"Elapsed time: {elapsed_time:.5}")

        else:
            
            env_list = []
            eval_list = []
            for level in curriculum_list:
                current_folder = args.out + "/model_outputs_" + level + str(map_ind)
                if not os.path.exists(current_folder):
                    os.makedirs(current_folder)
                
                if level == "easy":
                    current_map = np.copy(easy_map)
                elif level == "medium":
                    current_map = np.copy(medium_map)
                else:
                    current_map = np.copy(gen_map)

                # train_env = DummyVecEnv([lambda: AgentFormation(generated_map=current_map, map_lim=args.map_lim)])
                # train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False, clip_obs=1.)
                # train_env = VecMonitor(train_env, filename = current_folder + "/monitor.csv")
                # train_env = AgentFormation(generated_map=current_map, map_lim=args.map_lim, max_steps=250)
                # train_env = Monitor(train_env, current_folder + "/monitor.csv")
                train_env = make_vec_env(lambda: AgentFormation(generated_map=current_map, map_lim=args.map_lim), n_envs=args.n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
                env_list.append(train_env)

                eval_env = DummyVecEnv([lambda: AgentFormation(generated_map=current_map, map_lim=args.map_lim)])
                # eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False, clip_obs=1.)
                eval_list.append(eval_env)

            
            for index, level in enumerate(curriculum_list):
                print (f"\nCurriculum Level: {level}")
                
                env_list[index].reset()
                previous_index = np.clip(index-1, 0, len(curriculum_list) - 1)

                if index == 0:
                    model = model_def('MlpPolicy', env_list[0], verbose=0, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
                else:
                    model_name = "best_model_" + curriculum_list[previous_index] + str(map_ind)
                    model = PPO.load(model_dir + "/" +  model_name + "/best_model", verbose=1)
                    # model = PPO.load(model_dir + "/last_model_easy" + str(map_ind) +".zip", verbose=1)
                    print (model_name + " is loaded!")

                model.set_env(env_list[index])
                
                # eval_env = VecMonitor(eval_env, filename = current_folder + "/monitor.csv")
                callback = EvalCallback(eval_env=eval_list[index], eval_freq = N_eval // args.n_procs, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                                        best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=0)

                
                
                start = time.time()
                model.learn(total_timesteps=args.train_epochs, tb_log_name=model_name + "_run_" + level, callback=callback)
                model.save(model_dir + "/last_model_" + level + str(map_ind))
                # stats_path = os.path.join(model_dir, "vec_normalize.pkl")
                # env_list[index].save(stats_path)
                # eval_reward, _  = evaluate_policy(model, eval_list[index], n_eval_episodes=args.eval_episodes)
                elapsed_time = time.time() - start
                print (f"Elapsed time: {elapsed_time:.5}")
            

        train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--train_epochs', default=100000, type=int, help='number of test iterations')
    # parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    # parser.add_argument('--train_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--map_lim', default=10, type=int, help='width and height of the map')
    parser.add_argument('--n_procs', default=8, type=int, help='number of processes to execute')
    parser.add_argument('--n_maps', default=10, type=int, help='number of maps to train')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--out', default="output", type=str, help='the output folder')
    parser.add_argument('--map_file', default="saved_maps_10", type=str, help='the output folder')
    parser.add_argument('--nocurriculum', default = False, action='store_true', help='train on only the target map')
    args = parser.parse_args()
    
    args.out = "output_cur_10_100k"
    main(args)

    args.nocurriculum = True
    args.out = "output_nocur_10_100k"
    main(args)

