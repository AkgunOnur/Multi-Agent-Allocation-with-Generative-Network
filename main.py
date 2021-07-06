import os
import torch
import time
import argparse
import glob
import re
import pickle
import numpy as np

from dqn_model import *
from point_mass_formation import AgentFormation


def main():
    model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    # model_dir = './saved_models'
    # load_model_dir = './models'
    best_reward = 0
    final_reward = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = False

    # Create environments.
    env = AgentFormation(visualization=visualization)
    

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    parser.add_argument('--load_model', default=load_model_dir, help='number of training episodes')
    parser.add_argument('--test_iteration', default=50, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=15, type=int, help='seed number for test')
    parser.add_argument('--single_model', default=False, action='store_true', help='single model to evaluate')
    # training
    parser.add_argument('--num_episodes', default=5000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=1, help='number of steps to update the policy')
    parser.add_argument('--eval_interval', type=int, default=1000, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=0, help='After how many steps to start training')
    # model
    parser.add_argument('--model_dir', default=model_dir, help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='Batch size to train')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=20, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    parser.add_argument('--n_states', type=int, default=216, help='Number of states after convolution layer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train')
    parser.add_argument('--memory_size', type=int, default=100000, help='Buffer memory size')
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    parser.add_argument('--out_shape', type=int, default=env.out_shape, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=100, help='Hidden size dimension')
    parser.add_argument('--device', default=device, help='device')


    args = parser.parse_args()
    

    dqn = DQN(args)

    model_reward_list = {}


    if args.test:
        np.random.seed(args.seed)
        mean_reward = 0

        if args.single_model:
            dqn.load_models(os.path.join(args.load_model, 'best'), 1)

            for i_iter in range(args.test_iteration):
                agent_obs = env.reset()
                episode_reward = 0

                action = dqn.choose_action(agent_obs) # output is between 0 and 7
                n_agents = action + 2 # number of allowable agents is 2 to 9
                episode_reward, done, agent_next_obs = env.step(n_agents)

                print('Episode: ', i_iter, '| Episode_reward: ', round(episode_reward, 2))

                mean_reward += episode_reward

            mean_reward = mean_reward / args.test_iteration
        
        else:
            for name in glob.glob(os.path.join(args.load_model, '*.pth')):
                mean_reward = 0
                if name.find('policy') > 0:
                    model_no = (int(re.findall(r'\d+', name)[0]))
                    dqn.load_models(args.load_model, model_no)

                    for i_iter in range(args.test_iteration):
                        agent_obs = env.reset()
                        episode_reward = 0

                        action = dqn.choose_action(agent_obs) # output is between 0 and 7
                        n_agents = action + 2 # number of allowable agents is 2 to 9
                        episode_reward, done, agent_next_obs = env.step(n_agents)

                        print('Episode: ', i_iter, '| Episode_reward: ', round(episode_reward, 2))

                        mean_reward += episode_reward

                    mean_reward = mean_reward / args.test_iteration
                    print('Model: {0} / Mean Reward: {1:.3} \n'.format(model_no, mean_reward))
                    model_reward_list[model_no] = mean_reward

                    with open('model_reward_list.pkl', 'wb') as f:  
                        pickle.dump(model_reward_list, f)
            
    else:
        for i_episode in range(args.num_episodes):
            agent_obs = env.reset()
            episode_reward = 0

            action = dqn.choose_action(agent_obs) # output is between 0 and 7
            n_agents = action + 2 # number of allowable agents is 2 to 9
            episode_reward, done, agent_next_obs = env.step(n_agents)

            dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)

            if i_episode > args.start_step and i_episode % args.update_interval == 0:
                dqn.learn()

            if episode_reward > best_reward:
                best_reward = episode_reward
                dqn.save_models(os.path.join(args.model_dir, 'best'), 1)

            if i_episode % args.eval_interval == 0 and i_episode > args.start_step:
                mean_reward = 0
                for i_iter in range(args.test_iteration):
                    agent_obs = env.reset()
                    episode_reward = 0

                    action = dqn.choose_action(agent_obs) # output is between 0 and 7
                    n_agents = action + 2 # number of allowable agents is 2 to 9
                    episode_reward, done, agent_next_obs = env.step(n_agents)

                    mean_reward += episode_reward

                mean_reward = mean_reward / args.test_iteration
                print('E | Episode: ', i_episode + 1, '| Evaluation reward: ', round(mean_reward, 2), '\n')
                if mean_reward > final_reward:
                    final_reward = mean_reward
                    dqn.save_models(os.path.join(args.model_dir, 'final'), i_episode)

            print('T | Episode: ', i_episode + 1, '| Episode reward: ', round(episode_reward, 2))

    
    env.close()




if __name__ == '__main__':
    main()

