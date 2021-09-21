import argparse
import numpy as np

#from dqn_model import *

from point_mass_formation import AgentFormation
from read_maps import *


def parameters():
    # model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/saved_models'
    # load_model_dir = '/okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/models'
    model_dir = './saved_models'

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--device', default=device, help='device')
    parser.add_argument('--visualization', default=False, type=bool, help='number of training episodes')
    # test
    #parser.add_argument('--test', default=False, action='store_true', help='number of training episodes')
    #parser.add_argument('--test_iteration', default=25, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=7, type=int, help='seed number for test')
    #parser.add_argument('--test_model_no', default=0, help='single model to evaluate')
    #parser.add_argument('--test_model_level', default="easy", help='single model level to evaluate')
    # training
    #parser.add_argument('--num_episodes', default=1000000, type=int, help='number of training episodes')
    parser.add_argument('--update_interval', type=int, default=32, help='number of steps to update the policy')
    #parser.add_argument('--eval_interval', type=int, default=32, help='number of steps to eval the policy')
    parser.add_argument('--start_step', type=int, default=128, help='After how many steps to start training')
    # model
    #parser.add_argument('--resume', default=False, action='store_true', help='to continue the training')
    parser.add_argument('--model_dir', default='./saved_models', help='folder to save models')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epsilon', default=0.9, type=float, help='greedy policy')
    parser.add_argument('--gamma', default=0.99, type=float, help='reward discount')
    parser.add_argument('--target_update', default=48, type=int, help='target update freq')
    parser.add_argument('--n_actions', type=int, default=8, help='number of actions (agents to produce)')
    #parser.add_argument('--n_states', type=int, default=7350, help='Number of states after convolution layer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size to train')
    parser.add_argument('--memory_size', type=int, default=250000, help='Buffer memory size')
    parser.add_argument('--multi_step', type=int, default=1, help='Multi step')
    #parser.add_argument('--out_shape', type=int, default=10, help='Observation image shape')
    parser.add_argument('--hid_size', type=int, default=64, help='Hidden size dimension')
    parser.add_argument('--out_shape_list', type=list, default=[80], help='output shape array')#default=[20,40,60,80]
    parser.add_argument('--fc1_shape_list', type=list, default=[4096], help='fc1 size array')#default=[16,576,1936,4096]

    return parser.parse_args()

class env_class:
    def __init__(self, mode='train'):
        self.args = parameters()
        self.iteration = 0
        self.mode = mode

        self.args.out_shape = 1#TODO #self.args.out_shape_list[current_scale]
        self.args.fc1_size = 1#TODO: self.args.fc1_shape_list[current_scale]
        
        # Create environments.
        self.env = AgentFormation(visualization=self.args.visualization)

    def reset_and_step(self, ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list, n_agents):
        #ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

        #env.reset to initalize map for D* (obstacle_locations, prize_locations etc.)
        self.env.reset(ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list)
        #Step environment
        episode_reward, _, _ = self.env.step(n_agents)

        if self.args.visualization:
            self.env.close()

        return episode_reward