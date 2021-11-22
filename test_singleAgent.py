import pickle
import torch
import numpy as np
import torch.nn as nn
import gym
import time

from read_maps import fa_regenate
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.singleAgentTestEnv import TestQuadrotorFormation
from environment.level_utils import read_level
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from environment.level_utils import read_level, one_hot_to_ascii_level
from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS
from config import get_arguments, post_config
from construct_library import Test_Library

replace_tokens = REPLACE_TOKENS

opt = get_arguments().parse_args()
opt = post_config(opt)

test_lib = Test_Library()

def main(opt):

    vecenv = make_vec_env(lambda: TestQuadrotorFormation(), n_envs=1, vec_env_cls=SubprocVecEnv)
    #env = TestQuadrotorFormation()
    model = A2C.load("./weights/a2c_gan", env = vecenv)

    # with open('./train_lib/training_map_library.pkl', 'rb') as f:
    #     map_dataset = pickle.load(f)
    #     map_dataset = np.array(map_dataset[0]).squeeze(1) 

    # for i in range(15):
    #     opt.input_dir = "test_bench"
    #     opt.input_name = f"test{i+1}.txt"
    #     #opt.input_dir = "./test_bench"
    #     #opt.input_name = "test4.txt"
    #     #opt.input_dir = "./input/"
    #     #opt.input_name = "medium_map.txt"
    #     #opt.input_dir = "./output/generated_maps/hard"
    #     #opt.input_name = "hard-400.txt"
        

    #     test_map = read_level(opt, None, replace_tokens)
    #     test_lib.add(test_map)
        
    # test_lib.save_maps()

    total_rew = 0
    
    for i in range(15):

        done = False
        obs = vecenv.reset()
        print(obs.shape)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, rewards, done, info = vecenv.step(action)
            total_rew += rewards
            time.sleep(0.05)
    print(total_rew)
if __name__ == '__main__':
    main(opt)
