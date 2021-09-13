# Code inspired by https://github.com/tamarott/SinGAN
import os

import torch
import wandb
from tqdm import tqdm

from models import init_models, reset_grads, restore_weights
from environment.level_utils import load_level_from_text
from dqn_model import *
from point_mass_formation import AgentFormation
from rl_agent import rl
import os
import glob

def test(opt):
    """ Wrapper function for testubg. Get test maps and then calls test_single_scale on each. """

    opt.scales.insert(0, 1)

    test_map_dir = os.listdir(opt.test_dir)
    test_map_dir.sort()

    # Test Loop
    for directory in test_map_dir:
        current_dir = os.path.join(opt.test_dir, directory +"/")
        #print("current_dir: ", current_dir)
        file_names = glob.glob("./"+current_dir+"*.txt")

        scale_number = int(int(directory)/20-1)
        #print("scale_number: ", scale_number)

        #initalizerl agent and load its weights
        RL = rl(int(scale_number), 'test')

        agent_mean_reward = 0.0

        #for maps in this scale
        for i in range(len(file_names)):
            map = load_level_from_text(file_names[i])
            
            #Deploy agent in map and get reward for couple of iterations
            agent_mean_reward += RL.test(map)
            #print("Map: "+ str(file_names[i])+ " agent_mean_reward: ", agent_mean_reward)
            #log rewards
        print("agent_mean_reward for scale" + str(scale_number) + " is :", agent_mean_reward/len(file_names))
