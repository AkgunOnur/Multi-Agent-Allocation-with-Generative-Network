import sys
import torch
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng


import os
from point_mass_env import AgentFormation
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from train import generate_maps, curriculum_design
from utils import *

sys.path.insert(0,'rendering_files/')

from rendering_files.level_utils import load_level_from_text
from rendering_files.level_image_gen import LevelImageGen


def test_single_map():
    map_name = "map_0"
    main_folder = "output_Noncurriculum_8Haziran"
    folder_postfix = "8Haziran_PPO_Noncurriculum_5M_map_0"
    model_dir = main_folder + '/saved_models'
    visualization = True

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--eval_episodes', default=100, type=int, help='number of test iterations')
    parser.add_argument('--map_lim', default=20, type=int, help='width and height of the map')

    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)

    target_map = []
    lvl = load_level_from_text("target_maps/" + map_name + ".txt")
    current_map = np.zeros((args.map_lim, args.map_lim))
    for r in range(args.map_lim):
        for c in range(args.map_lim):
            if lvl[r][c] == '-':
                current_map[r][c] = 0
            elif lvl[r][c] == 'W':
                current_map[r][c] = 1
            elif lvl[r][c] == 'X':
                current_map[r][c] = 2
    target_map.append(current_map)

    model = PPO.load(model_dir + "/" + folder_postfix + "/best_model", verbose=1) # + "/best_model"
    env = AgentFormation(generated_map=target_map, map_lim=args.map_lim, max_steps=20*args.map_lim, visualization=visualization)
    print ("\nBest model for map is loaded!")
    map_reward_list = []
    # episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes,
    #         render=False,
    #         deterministic=False,
    #         return_episode_rewards=True,
    #         warn=True)
    # print ("Episode Rewards: ", episode_rewards, " Episode length: ", episode_lengths, "Mean reward: ", np.mean(episode_rewards), "Mean length: ", np.mean(episode_lengths))

    for episode in range(args.eval_episodes):
        obs = env.reset()
        map_reward = 0
        iteration = 0
        iter_list = []
        while True:
            action, _states = model.predict(obs, deterministic = False)
            print ("action:", action)
            obs, reward, done, info = env.step(action)
            # print ("reward: ", reward)
            map_reward += reward
            iteration += 1
        
            if done:
                print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, map_reward, iteration))
                map_reward_list.append(map_reward)
                iter_list.append(iteration)
                break
    
    print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(map_reward_list), np.std(map_reward_list), args.eval_episodes))

    
    # with open('results_' + file + '.pickle', 'wb') as handle:
    #     pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if visualization:
        env.close()

def test_multi_maps():
    level = "level5"
    file = "output_cur_basit_20_20k"
    model_dir = file + '/saved_models'
    visualization = True
    agent_step_cost = 0.01
    curriculum_list = ["level1", "level2", "level3", "level4", "level5"]
    index = 2

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--eval_episodes', default=5, type=int, help='number of test iterations')
    parser.add_argument('--map_file', default="", type=str, help='the map file')
    parser.add_argument('--map_lim', default=20, type=int, help='width and height of the map')

    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)

    current_map_size = args.map_lim
    target_map_lim = args.map_lim

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    # with open('saved_maps_' + str(map_lim) + '.pickle', 'rb') as handle:
    #     easy_list, medium_list, gen_list = pickle.load(handle)
    
    with open(args.map_file + '.pickle', 'rb') as handle:
        generated_map_list = pickle.load(handle)

    
    folder_list = [x[0].split('/')[1] for x in os.walk(file + "/") if x[0][-5:] == ("map_" + str(index)) and x[0].split('/')[1].split('_')[0] == "train"]
    order_list = [int(folder.split('_')[1]) for folder in folder_list]
    sorted_folders = [folder for order, folder in sorted(zip(order_list, folder_list))]

    for folder in sorted_folders[::-1]:
        if folder.split('_')[2] == "level5":
            model_folder = folder
            break

    total_reward_list = []
    # for index in range(1,10):
    map_list = []
    map_list.append(generated_map_list[curriculum_list[-1]][index])
    N_reward = len(np.argwhere(map_list[0] == 2))
    max_possible_reward = N_reward - agent_step_cost * 100
    model = PPO.load(model_dir + "/best_model" + model_folder[5:] + "/best_model", verbose=1) # + "/best_model"
    env = AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=visualization)
    print ("\nBest model for map ", index, " is loaded!")
    map_reward_list = []
    # episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes,
    #         render=False,
    #         deterministic=False,
    #         return_episode_rewards=True,
    #         warn=True)
    # print ("Episode Rewards: ", episode_rewards, " Episode length: ", episode_lengths, "Mean reward: ", np.mean(episode_rewards), "Mean length: ", np.mean(episode_lengths))

    for episode in range(args.eval_episodes):
        obs = env.reset()
        map_reward = 0
        iteration = 0
        iter_list = []
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print ("reward: ", reward)
            map_reward += reward
            iteration += 1
        
            if done:
                print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, map_reward, iteration))
                map_reward_list.append(map_reward)
                iter_list.append(iteration)
                break
    
    total_reward_list.append(map_reward_list)
    print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(map_reward_list), np.std(map_reward_list), args.eval_episodes))

    
    # with open('results_' + file + '.pickle', 'wb') as handle:
    #     pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if visualization:
        env.close()

if __name__ == '__main__':
    test_single_map()

