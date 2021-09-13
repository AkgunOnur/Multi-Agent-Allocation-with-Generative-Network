import os
import torch
import time
import argparse
import glob
import re
import pickle
import numpy as np
import csv
from collections import defaultdict


from dqn_model import *

from point_mass_formation import AgentFormation


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = False

    # Create environments.
    env = AgentFormation(visualization=visualization)
    
    parser = argparse.ArgumentParser(description='Collect dataset for classifier')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--num_episodes', default=600, type=int, help='number of episodes')
    parser.add_argument('--resume', default=False, action='store_true', help='to continue the training')
    parser.add_argument('--device', default=device, help='device')


    args = parser.parse_args()
    
    model_reward_list = {}

    map_state_list = []
    best_option_list = []
    # time.sleep(0.5)

    for i_episode in range(1, args.num_episodes + 1):

        if i_episode <= 100:
            level = "easy"
            # index = 0
        elif i_episode > 100 and i_episode <= 300:
            level = "medium"
            # index = np.random.choice(6)
        else:
            level = "hard"
            # index = np.random.choice(3)
        
        seed_no = np.random.randint(1500)
        reward_list = defaultdict(list)
        for n_agents in range(1,9):
            np.random.seed(seed_no)
            map_state = env.reset(level)
            if n_agents == 1:
                map_state_list.append(map_state)
            episode_reward, done, agent_next_obs = env.step(n_agents)
            reward_list[n_agents] = episode_reward

            print('Data Collection - ', level,' | Episode: ', i_episode, ' | n_agents: ', n_agents, '| Episode reward: ', round(episode_reward, 2))

        best_option = max(reward_list, key=reward_list.get)
        best_option_list.append(best_option)

        if visualization:
            env.close()

        with open('dataset.pickle', 'wb') as handle:
            pickle.dump([map_state_list, best_option_list], handle, protocol=pickle.HIGHEST_PROTOCOL)

    if visualization:
        env.close()

if __name__ == '__main__':
    main()

