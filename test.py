import sys
import torch
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng


import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym_minigrid.envs.custom_doorkey import CustomDoorKeyEnv
from env_util import make_vec_env, make_atari_env, RGBImgObsWrapper



def test():
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--eval_episodes', default=3, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=1, type=int, help='number of seed')
    parser.add_argument('--visualization', default=True, type=bool, help='vis')
    parser.add_argument('--env_type', default="DoorKey", type=str, help='scenario') # LavaGap, KeyCorridor
    args = parser.parse_args()
    
    map_reward_list = []
    target_index = 2
    model_dir = "3Ekim_DoorKey_iter_0_map_" + str(target_index) + "/saved_models"
    
    
    if args.env_type == "KeyCorridor":
        env_id = "MiniGrid-CustomKeyCorridor-v0"
        with open("keycorridor_all_maps.pickle", 'rb') as handle:
            map_lists = pickle.load(handle)

    elif args.env_type == "LavaGap":
        env_id = "MiniGrid-CustomLavaGap-v0"
        with open("target_maps_lavagap.pickle", 'rb') as handle:
            map_lists = pickle.load(handle)

    elif args.env_type == "DoorKey":
        env_id = "MiniGrid-CustomDoorKey-v0"
        with open("doorkey_all_maps.pickle", 'rb') as handle:
            map_lists = pickle.load(handle)
        

    for episode in range(args.eval_episodes):
        current_map = map_lists[target_index]
        env_kwargs = {"visualization": args.visualization, "env_map": current_map}    # inner_map_lim = 6
        env = make_atari_env(env_id=env_id, n_envs=1, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)
        model = PPO.load(model_dir + "/best_model", verbose=1) # + "/best_model"
        # env = RGBImgObsWrapper(CustomDoorKeyEnv(config))
        # env = make_vec_env(env_id=env_id, n_envs=1, env_kwargs=dict(config=config,frame_file = "frames_5"), vec_env_cls=DummyVecEnv)

        obs = env.reset()
        map_reward = 0
        iteration = 0
        iter_list = []
        while True:
            action, _states = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)
            # print ("reward: ", reward)
            map_reward += reward[0]
            iteration += 1
            time.sleep(0.1)
        
            if done:
                print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, map_reward, iteration))
                map_reward_list.append(map_reward)
                iter_list.append(iteration)
                break
    
    print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(map_reward_list), np.std(map_reward_list), args.eval_episodes))

    
    # with open('results_' + file + '.pickle', 'wb') as handle:
    #     pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.visualization:
        env.close()

if __name__ == '__main__':
    test()

