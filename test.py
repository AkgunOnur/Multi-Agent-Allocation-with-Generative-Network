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



def main(args):
    map_lim = 10

    if args.mode == 1:
        typ = "cur"
    elif args.mode == 0:
        typ = "nocur"
    else:
        print("fatal error ?")

    model_dir = 'output_' + typ + '_' + str(map_lim) + '/saved_models'

    visualization = args.visualization


    os.makedirs(model_dir, exist_ok=True)

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    with open('saved_maps_' + str(map_lim) + '.pickle', 'rb') as handle:
        easy_list, medium_list, gen_list = pickle.load(handle)


    total_reward_list = []
    total_iter_list = []
    
    for index in range(10):
        model = PPO.load(model_dir + "/last_model_" + "target" + str(index), verbose=1) # + "/best_model"
        env = AgentFormation(generated_map=gen_list[index], map_lim=map_lim, visualization=visualization, max_steps=1000)
        print ("\nBest model for map ", index, " is loaded!")
        
        map_reward_list = []
        map_iter_list = []
        
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
            
            while True:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                
                if visualization:
                    print ("reward: ", reward)
                    time.sleep(0.1)
                
                map_reward += reward
                iteration += 1
            
                if done:
                    print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, map_reward, iteration))
                    map_reward_list.append(map_reward)
                    map_iter_list.append(iteration)
                    break
        
        total_reward_list.append(map_reward_list)
        total_iter_list.append(map_iter_list)

        print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(map_reward_list), np.std(map_reward_list), args.eval_episodes))

        
        with open('results_' + typ + '_' + str(map_lim) + '.pickle', 'wb') as handle:
            pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('iterations_' + typ + '_' + str(map_lim) + '.pickle', 'wb') as handle:
            pickle.dump(total_iter_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if visualization:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL trainer')
    
    parser.add_argument('--mode', default=0, type=int, help='select mode {nocur: 0, cur: 1}')
    parser.add_argument('--visualization', default=0, type=int, help='enable visualization')
    parser.add_argument('--eval_episodes', default=10, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='seed number for test')

    args = parser.parse_args()

    main(args)

