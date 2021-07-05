import os
import torch
import time
import numpy as np

from dqn_model import *
from point_mass_formation import AgentFormation


def main():
    N_episodes = 1000
    update_interval = 1
    start_step = 0
    eval_interval = 5
    model_dir = './models'
    best_reward = 0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visualization = True

    # Create environments.
    env = AgentFormation(visualization=visualization)
    dqn = DQN()

    for i_episode in range(N_episodes):
        agent_obs = env.reset()
        episode_reward = 0

        action = dqn.choose_action(agent_obs) # output is between 0 and 7
        n_agents = action + 2 # number of allowable agents is 2 to 9
        episode_reward, done, agent_next_obs = env.step(n_agents) # next observation haritanin son durumunu gostersin

        dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)

        if i_episode > start_step and i_episode % update_interval == 0:
            dqn.learn()

        if episode_reward > best_reward:
            best_reward = episode_reward
            dqn.save_models(os.path.join(model_dir, 'best'), 1)

        if i_episode % eval_interval == 0 and i_episode >= start_step:
            dqn.save_models(os.path.join(model_dir, 'final'), i_episode)

        print('Episode: ', i_episode, '| Episode_reward: ', round(episode_reward, 2))




if __name__ == '__main__':
    main()
