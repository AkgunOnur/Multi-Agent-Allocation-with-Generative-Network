import os
import torch
import time
import argparse
import glob
import re
import pickle
import numpy as np
import csv

from dqn_model import *

from point_mass_formation import AgentFormation


def main():
    model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    # model_dir = './saved_models'
    # load_model_dir = './models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = False

    # Create environments.
    env = AgentFormation(visualization=visualization)
    

    parser = argparse.ArgumentParser(description='RL trainer')
    # test
    parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    parser.add_argument('--load_model', default=load_model_dir, help='number of training episodes')
    parser.add_argument('--test_iteration', default=25, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    parser.add_argument('--test_model_no', default=1, help='single model to evaluate')
    # training
    parser.add_argument('--num_episodes', default=7000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=10, help='number of steps to update the policy')
    parser.add_argument('--eval_interval', type=int, default=50, help='number of steps to eval the policy')
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
    parser.add_argument('--memory_size', type=int, default=1000000, help='Buffer memory size')
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    parser.add_argument('--out_shape', type=int, default=env.out_shape, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=100, help='Hidden size dimension')
    parser.add_argument('--device', default=device, help='device')


    args = parser.parse_args()
    

    dqn = DQN(args)

    model_reward_list = {}

    level_list = ["easy", "medium", "hard"]
    level_rewards = {"easy":-100, "medium":-100, "hard":-100}
    fields = ["Model", "Level", "Mean Reward", "Total Episodes"]
    filename = "test_results.txt"
    test_info = []
    if args.test:
        print ("Test Mode!")
        # time.sleep(0.5)
        # np.random.seed(args.seed)

        if int(args.test_model_no) > 0:
            model_path = dqn.load_models(args.load_model, "hard", args.test_model_no)
        elif int(args.test_model_no) == 0:
            print ("Random policy")
            model_path = "Random"
        else:
            print ("Wrong input!")
            return 

        for level in level_list:
            mean_reward = 0
            index = 0

            for i_iter in range(1, args.test_iteration + 1):

                if level == "medium":
                    index = np.random.choice(6)
                elif level == "hard":
                    index = np.random.choice(3)
                
                agent_obs = env.reset(level, index)
                episode_reward = 0
                action = dqn.choose_action(agent_obs) # output is between 0 and 7
                n_agents = action + 1 # number of allowable agents is 1 to 8
                episode_reward, done, agent_next_obs = env.step(n_agents)
                print('Test - ', level,' | Episode: ', i_iter, '| Episode reward: ', round(episode_reward, 2))

                if visualization:
                    env.close()

                mean_reward += episode_reward

            mean_reward = mean_reward / args.test_iteration
            level_rewards[level] = mean_reward
            print('Test - ', level, ' | Model: ', model_path, ' | Mean reward: ', round(mean_reward, 2))
            test_info.append([model_path, level, round(mean_reward, 2), args.test_iteration])

        mean_levels = np.array(list(level_rewards.values())).mean()
        test_info.append([model_path, "Mean", round(mean_levels, 2), args.test_iteration])
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter='|')
            # csvwriter.writerow(fields) 
            csvwriter.writerows(test_info)
            
    else:
        print ("Train Mode!")
        level = "easy"
        previous_mode = False
        # time.sleep(0.5)
        for i_episode in range(1, args.num_episodes + 1):
            if i_episode % 5 == 0 and level != "easy":
                previous_mode = True

            if previous_mode == False:
                if i_episode <= 1e6:
                    level = "easy"
                    # index = 0
                elif i_episode > 1e6 and i_episode <= 3e6:
                    level = "medium"
                    # index = np.random.choice(6)
                else:
                    level = "hard"
                    # index = np.random.choice(3)
                level_actual = level
            else:
                ind = np.random.randint(level_list.index(level))
                level = level_list[ind]
                previous_mode = False

            agent_obs = env.reset(level)
            episode_reward = 0

            action = dqn.choose_action(agent_obs) # output is between 0 and 7
            n_agents = action + 1 # number of allowable agents is 1 to 8
            episode_reward, done, agent_next_obs = env.step(n_agents)

            if visualization:
                env.close()

            dqn.memory.append(agent_obs, action, episode_reward, agent_next_obs, done)

            if i_episode > args.start_step and i_episode % args.update_interval == 0:
                dqn.learn()

            # if episode_reward > level_rewards[level]:
            #     train_reward = episode_reward
            #     dqn.save_models(os.path.join(args.model_dir, 'train'), "model", 1)
            
            if i_episode % 100 == 0:
                print('Train - ', level,' | Episode: ', i_episode, '| Episode reward: ', round(episode_reward, 2))

            if i_episode % args.eval_interval == 0 and i_episode > args.start_step and previous_mode == False:
                mean_reward = 0
                for i_iter in range(args.test_iteration):
                    # manual curriculum
                    # index = 0
                    # if level == "medium":
                    #     index = np.random.choice(6)
                    # elif level == "hard":
                    #     index = np.random.choice(3)

                    agent_obs = env.reset(level_actual)
                    episode_reward = 0

                    action = dqn.choose_action(agent_obs) # output is between 0 and 7
                    n_agents = action + 1 # number of allowable agents is 1 to 8
                    episode_reward, done, agent_next_obs = env.step(n_agents)

                    if visualization:
                        env.close()

                    mean_reward += episode_reward

                mean_reward = mean_reward / args.test_iteration
                print('Eval - ', level_actual,' | Episode: ', i_episode, '| Evaluation reward: ', round(mean_reward, 2), '\n')
                if mean_reward > level_rewards[level_actual]:
                    level_rewards[level_actual] = mean_reward
                    dqn.save_models(os.path.join(args.model_dir, 'eval'), level_actual, i_episode)

    if visualization:
        env.close()

if __name__ == '__main__':
    main()

