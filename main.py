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
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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



def main():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = 'saved_models'
    load_model_dir = 'models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = True

    # Create environments.
    # env = AgentFormation(visualization=visualization)
    
    

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
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
    
    # Logs will be saved in model_dir/monitor.csv
    # env_monitor = Monitor(env, model_dir)

    if args.test:
        print ("Test mode!")
        env = AgentFormation(generated_map=easy_map, visualization=visualization, max_steps=1000)
        model = A2C.load(model_dir + "/best_model.zip", verbose=1)
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

