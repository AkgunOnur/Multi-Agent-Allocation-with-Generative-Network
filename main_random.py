from loguru import logger
from tqdm import tqdm
from config import get_arguments, post_config


from rl_agent import rl
from generate_random_map import generate_random_map

import pandas as pd
import os

reward_columns = ['total_reward']
def write_reward(stats,reward_filename):
    df_stats = pd.DataFrame([stats], columns=reward_columns)
    df_stats.to_csv(reward_filename, mode='a', index=False,header=not os.path.isfile(reward_filename))

def main(opt):
    """ Train RL Agent for all four scales one by one with random generated maps"""
    opt.scales.insert(0, 1)
    for scale_number, scale_value in reversed(list(enumerate(opt.scales))):
        scale_number = abs(scale_number-len(opt.scales)+1)
        RL = rl(scale_number)
        reward_filename = "reward"+ str(scale_number)+ ".csv" # path to CSV file
    
        # #Increase niter two times at final layer
        # if(scale_value==len(reals)-1):
        #     opt.niter = opt.niter*2

        for epoch in tqdm(range(opt.niter)):
            #Generate fake map(s) and make it playable
            coded_fake_map = generate_random_map(int(scale_value*opt.full_map_size)) #map size = scale_value*opt.full_map_size

            #Deploy agent in map and get reward for couple of iterations
            agent_mean_reward = RL.train_random(coded_fake_map, epoch)
            write_reward([agent_mean_reward], reward_filename)
        # Save networks
        RL.dqn.save_models(str(scale_number)+'_last')

if __name__ == "__main__":
    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)
    main(opt)