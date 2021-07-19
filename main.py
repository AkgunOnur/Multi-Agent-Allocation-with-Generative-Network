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
    train_reward = -1e3
    eval_reward = -1e3

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
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--test_model_no', default=-1, help='single model to evaluate')
    # training
    parser.add_argument('--num_episodes', default=7000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='number of steps to update the policy')
    parser.add_argument('--eval_interval', type=int, default=500, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=0, help='After how many steps to start training')
    # model
    parser.add_argument('--model_dir', default=model_dir, help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='Batch size to train')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=20, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    parser.add_argument('--n_states', type=int, default=150, help='Number of states after convolution layer')
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
        level = "hard"
        index = 0
        if int(args.test_model_no) > 0:
            dqn.load_models(args.load_model, args.test_model_no)

            for i_iter in range(args.test_iteration*1, args.test_iteration*5):
                if level == "medium":
                    index = np.random.choice(6)
                elif level == "hard":
                    index = np.random.choice(3)

                agent_obs = env.reset(level, index)
                episode_reward = 0
                action = dqn.choose_action(agent_obs) # output is between 0 and 7
                n_agents = action + 1 # number of allowable agents is 1 to 8
                episode_reward, done, agent_next_obs = env.step(n_agents)

                print('Episode: ', i_iter + 1, '| Episode Reward: ', round(episode_reward, 2))

                mean_reward += episode_reward

            mean_reward = mean_reward / args.test_iteration
            print('Model: {0} / Mean Reward: {1:.3} \n'.format(args.test_model_no, mean_reward))
        
        else:
            print ("Wrong input!")
            
    else:
        print ("Train Mode!")
        time.sleep(0.5)
        for i_episode in range(1, args.num_episodes + 1):
            if i_episode <= 1e6:
                level = "easy"
                index = 0
            elif i_episode > 1e6 and i_episode <= 3e6:
                level = "medium"
                index = np.random.choice(6)
            else:
                level = "hard"
                index = np.random.choice(3)

            agent_obs = env.reset(level, index)
            episode_reward = 0

            action = dqn.choose_action(agent_obs) # output is between 0 and 7
            n_agents = action + 1 # number of allowable agents is 1 to 8
            episode_reward, done, agent_next_obs = env.step(n_agents)

            if visualization:
                env.close()

            dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)

            if i_episode > args.start_step and i_episode % args.update_interval == 0:
                dqn.learn()

            if episode_reward > train_reward:
                train_reward = episode_reward
                dqn.save_models(os.path.join(args.model_dir, 'train'), level, 1)
            
            if i_episode % 100 == 0:
                print('Train - ', level,' | Episode: ', i_episode, '| Episode reward: ', round(episode_reward, 2))

            if i_episode % args.eval_interval == 0 and i_episode > args.start_step:
                mean_reward = 0
                for i_iter in range(args.test_iteration):
                    index = 0
                    if level == "medium":
                        index = np.random.choice(6)
                    elif level == "hard":
                        index = np.random.choice(3)

                    agent_obs = env.reset(level, index)
                    episode_reward = 0

                    action = dqn.choose_action(agent_obs) # output is between 0 and 7
                    n_agents = action + 1 # number of allowable agents is 1 to 8
                    episode_reward, done, agent_next_obs = env.step(n_agents)

                    if visualization:
                        env.close()

                    mean_reward += episode_reward

                mean_reward = mean_reward / args.test_iteration
                print('Eval - ', level ,' | Episode: ', i_episode, '| Evaluation reward: ', round(mean_reward, 2), '\n')
                if mean_reward > eval_reward:
                    eval_reward = mean_reward
                    dqn.save_models(os.path.join(args.model_dir, 'eval'), level, i_episode)

            

    
    if visualization:
        env.close()




if __name__ == '__main__':
    main()

