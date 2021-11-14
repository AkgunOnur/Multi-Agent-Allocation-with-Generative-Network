import torch
import time
import argparse
import numpy as np

import os
from point_mass_env import AgentFormation
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from utils import *



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
    parser.add_argument('--load_model', default=load_model_dir, help='number of training episodes')
    parser.add_argument('--test_iteration', default=25, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--test_model_no', default=0, help='single model to evaluate')
    parser.add_argument('--test_model_level', default="easy", help='single model level to evaluate')
    # training
    parser.add_argument('--num_episodes', default=1000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='number of steps to update the policy')
    parser.add_argument('--eval_interval', type=int, default=50, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=0, help='After how many steps to start training')
    # model
    parser.add_argument('--resume', default=False, action='store_true', help='to continue the training')
    parser.add_argument('--model_dir', default=model_dir, help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='Batch size to train')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=20, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    parser.add_argument('--n_states', type=int, default=7350, help='Number of states after convolution layer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train')
    parser.add_argument('--memory_size', type=int, default=100000, help='Buffer memory size')
    parser.add_argument('--device', default=device, help='device')


    args = parser.parse_args()
    os.makedirs(model_dir, exist_ok=True)
    reward_range = [-100, 10]

    num_cpu = 4

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], reward_range=reward_range)
    
    np.random.seed(7)
    gen_map = np.random.choice(3, (10,10), p=[0.75, 0.2, 0.05])
    
    env = AgentFormation(generated_map=gen_map, visualization=visualization)
    # Logs will be saved in model_dir/monitor.csv
    # env_monitor = Monitor(env, model_dir)

    if args.test:
        print ("Test mode!")
        env = AgentFormation(generated_map=gen_map, visualization=visualization)
        model = A2C.load('saved_models/82_11.zip')
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
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=model_dir, subfolder_dir="82_11")
        model = A2C('MlpPolicy', env, verbose=0).learn(total_timesteps=5000, callback=callback)
    

if __name__ == '__main__':
    main()

