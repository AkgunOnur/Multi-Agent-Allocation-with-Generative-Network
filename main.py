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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from train import generate_maps, curriculum_design
from utils import *



def main():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = 'output_cur/saved_models'
    load_model_dir = 'models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = True

    # Create environments.
    # env = AgentFormation(visualization=visualization)
    
    

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    parser.add_argument('--mode', default="target", help='which map to be executed')
    parser.add_argument('--train_steps', default=5000, type=int, help='number of test iterations')
    parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--train_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--n_procs', default=8, type=int, help='seed number for test')



    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)
    reward_range = [-100, 10]

    num_cpu = 4
    test_index = 30
    index = 217

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    # with open('generated_maps.pickle', 'rb') as handle:
    #     generated_maps = pickle.load(handle) 
    
    gen_map = generate_maps(seed=7)
    easy_map = curriculum_design(gen_map, level = "easy")
    medium_map = curriculum_design(gen_map, level = "medium")
    
    # Logs will be saved in model_dir/monitor.csv
    # env_monitor = Monitor(env, model_dir)

    if args.test:
        print (f"Test mode for {args.mode}")
        if args.mode == "easy":
            current_map = np.copy(easy_map)
        elif args.mode == "medium":
            current_map = np.copy(medium_map)
        elif args.mode == "target":
            current_map = np.copy(gen_map)
        else:
            print ("Invalid argument")
            return
            
        model = PPO.load(model_dir + "/best_model_" + "easy" + "/best_model", verbose=1)
        env = AgentFormation(generated_map=current_map, visualization=visualization, max_steps=1000)
        
        total_reward_list = []
        N_episode = 10
        for i in range(N_episode):
            obs = env.reset()
            total_reward = 0
            while True:
                action, _states = model.predict(obs)
                # action = np.random.randint(0,8)
                obs, reward, done, info = env.step(action)
                # print ("reward: ", reward)
                # time.sleep(0.1)
                total_reward += reward
            
                if done:
                    print ("Reward: {0:.3f} - Done: {1}".format(total_reward, done))
                    total_reward_list.append(total_reward)
                    break

        print ("Mean reward: {0:.3f} in {1} episodes".format(np.mean(total_reward_list), N_episode))
        if visualization:
            env.close()
    else:
        
        # gen_map = generated_maps[index]
        env = AgentFormation(generated_map=gen_map, visualization=visualization)
        model = A2C('MlpPolicy', env, verbose=0)
        for i in range(args.train_episodes):
            total_procs = args.n_procs + i
            # train_env = SubprocVecEnv([make_env(total_procs, gen_map) for j in range(args.n_procs)])
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=model_dir, subfolder_dir="map_" + str(index))
            model.learn(total_timesteps=5000, callback=callback)
        if visualization:
            env.close()
    

if __name__ == '__main__':
    main()

