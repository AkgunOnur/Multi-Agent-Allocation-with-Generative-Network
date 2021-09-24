import argparse
import numpy as np

#from dqn_model import *

from point_mass_formation import AgentFormation
from read_maps import *

class env_class:
    def __init__(self, mode='train'):
        self.iteration = 0
        self.mode = mode
        
        # Create environments.
        self.env = AgentFormation(visualization=False)

    def reset_and_step(self, ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list, n_agents):
        #ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

        #env.reset to initalize map for D* (obstacle_locations, prize_locations etc.)
        self.env.reset(ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list)
        #Step environment
        episode_reward, _, _ = self.env.step(n_agents)

        if self.args.visualization:
            self.env.close()

        return episode_reward