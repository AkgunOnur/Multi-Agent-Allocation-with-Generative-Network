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



def main():
    level = "level4"
    file = "output_cur_40_400k"
    model_dir = file + '/saved_models'
    visualization = True
    curriculum_list = ["level1", "level2", "level3", "level4"]

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--eval_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='seed number for test')
    parser.add_argument('--map_file', default="new_saved_maps_40", type=str, help='the map file')
    parser.add_argument('--map_lim', default=40, type=int, help='width and height of the map')

    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    # with open('saved_maps_' + str(map_lim) + '.pickle', 'rb') as handle:
    #     easy_list, medium_list, gen_list = pickle.load(handle)
    with open(args.map_file + '.pickle', 'rb') as handle:
        generated_map_list = pickle.load(handle)


    total_reward_list = []
    for index in range(1,10):
        map_list = []
        map_list.append(generated_map_list[curriculum_list[-1]][index])
        N_reward = len(np.argwhere(map_list[0] == 2))
        max_possible_reward = N_reward*10 - 0.15*N_reward*args.map_lim
        model = PPO.load(model_dir + "/best_model_" + level + str(index) + "/best_model", verbose=1) # + "/best_model"
        env = AgentFormation(generated_map=map_list, map_lim=args.map_lim, target_lim=args.map_lim, max_reward=max_possible_reward, max_steps=20*args.map_lim, visualization=visualization)
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

        
        with open('results_' + file + '.pickle', 'wb') as handle:
            pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if visualization:
            env.close()

if __name__ == '__main__':
    main()

