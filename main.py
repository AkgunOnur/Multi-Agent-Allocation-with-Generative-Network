import torch
import time
import argparse
import numpy as np

from point_mass_env import AgentFormation


def main():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = './saved_models'
    load_model_dir = './models'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    visualization = True

    # Create environments.
    env = AgentFormation(visualization=visualization)
    

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
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    parser.add_argument('--out_shape', type=int, default=env.out_shape, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=100, help='Hidden size dimension')
    parser.add_argument('--device', default=device, help='device')


    args = parser.parse_args()

    
    print ("Train Mode!")
    agent_obs = env.reset()
    episode_reward = 0
    for i_episode in range(1, args.num_episodes + 1):
        action = np.random.choice(8) # output is between 0 and 7
        reward, done, agent_next_obs = env.step(action)

        episode_reward += reward
        time.sleep(0.1)

        if i_episode % 1 == 0:
            print('Train - ',' | Episode: ', i_episode, '| Episode reward: ', round(episode_reward, 2))

        if done:
            print ("Mission completed!")
            break

    if visualization:
        env.close()

if __name__ == '__main__':
    main()

