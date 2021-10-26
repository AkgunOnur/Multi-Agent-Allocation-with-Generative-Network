import gym
from gym import spaces, error, utils
from gym.utils import seeding
#from gym.envs.classic_control import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from quadrotor_dynamics import Quadrotor, Drone, Bot
from numpy.random import uniform
from time import sleep
from collections import deque

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, visualization=False):

        self.seed()
        self.n_action = 6
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = n_agents
        self.visualization = visualization

        self.quadrotors = []
        self.viewer = None
        self.dtau = 1e-3

        self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit
        self.z_lim = 3

        self.agent_Xposition = 2
        self.agent_Yposition = 2

        self.map = np.zeros((3,20,20))
        self.wall_map = np.zeros(self.x_lim, self.y_lim)
        self.reward_map = np.zeros(self.x_lim, self.y_lim)
        self.path_map = np.zeros(self.x_lim, self.y_lim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, iteration, is_centralized):
        # Environment Step
        self.iteration = iteration
        done = False
        reward_list = np.ones(self.n_agents) * (-2)

        if is_centralized:
            agents_actions = self.action_list[action]
        else:
            agents_actions = np.reshape(action[0], (self.n_agents,))
            

        for agent_ind in range(self.n_agents):
            current_action = agents_actions[agent_ind]
            drone_current_state = self.get_drone_des_grid(agent_ind, current_action)

        if self.moving_target:
            for bot_ind in range(self.n_bots):
                if self.bots[bot_ind].is_alive:
                    self.get_bot_des_grid(bot_ind)
                
                    if np.linalg.norm(self.bots[bot_ind].state-self.bots[bot_ind].target_state) < 3.5:
                        target_pos = [self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-31, high=31), self.np_random.uniform(low=-12, high=-2)]
                        self.bots[bot_ind].target_state = target_pos


        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):
                if agent_ind != other_agents_ind:
                    collision_distance = np.linalg.norm(self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (collision_distance <= 7) and self.quadrotors[agent_ind].is_alive and self.quadrotors[other_agents_ind].is_alive:
                        done = True
                        self.quadrotors[agent_ind].is_alive = False
                        self.quadrotors[other_agents_ind].is_alive = False
                        self.quadrotors[agent_ind].state[2] = 0.0
                        self.quadrotors[other_agents_ind].state[2] = 0.0
                        reward_list[agent_ind] -= 300
                        reward_list[other_agents_ind] -= 300
            
            if not done:
                for bot_ind in range(self.n_bots):
                    drone_distance = np.linalg.norm(self.quadrotors[agent_ind].state-self.bots[bot_ind].state)
                    
                    if drone_distance <= 7 and self.bots[bot_ind].is_alive:
                        reward_list[agent_ind] += 100
                        self.bots[bot_ind].is_alive = False
                        self.bots[bot_ind].state[2] = 0.0


        if (not self.bots[0].is_alive) and (not self.bots[1].is_alive):
            done = True
            """
			for agent_ind in range(self.n_agents):
				reward_list[agent_ind] += 25
			"""

        if self.visualization:
            self.visualize()

        return self.get_observation(), reward_list/100, done, {}, [self.quadrotors[i].state for i in range(self.n_agents)], [self.bots[j].state for j in range(self.n_bots)]

    def get_observation(self):

        state = np.zeros((self.n_agents,self.n_agents*3+self.n_bots*3))

        for agent_ind in range(self.n_agents):
            state[agent_ind][0:3] = self.quadrotors[agent_ind].state

        return np.array(state)


    def generate_agent_position(self, agentX, agentY):
        self.quadrotors = []

        for idx in range(0, self.n_agents):
            state0 = [agentX[idx]/self.x_lim, agentY[idx]/self.y_lim]

            self.quadrotors.append(Drone(state0))

    def check_collision(self):
        collision = False
        for agent_ind in range(self.n_agents):
            for other_agents_ind in range(self.n_agents):

                if agent_ind != other_agents_ind:
                    dist = np.linalg.norm(self.quadrotors[agent_ind].state-self.quadrotors[other_agents_ind].state)

                    if (dist <= 7):
                        collision = True

        for bot_ind in range(self.n_bots):
            for other_bots_ind in range(self.n_bots):

                if bot_ind != other_bots_ind:
                    dist = np.linalg.norm(self.bots[bot_ind].state-self.bots[other_bots_ind].state)

                    if (dist <= 7):
                        collision = True

        return collision

    def reset(self, observation_map):
        self.path_map = observation_map[0]
        self.wall_map = observation_map[1]
        self.reward_map = observation_map[2]
        
        self.generate_agent_position(self.agent_Xposition, self.agent_Yposition)
        self.iteration = 1

        return self.get_observation()


    def get_bot_des_grid(self, bot_index):

        if self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] > 2:
            self.bots[bot_index].state[0] -= 0.3
        elif self.bots[bot_index].state[0] - self.bots[bot_index].target_state[0] < -2:
            self.bots[bot_index].state[0] += 0.3

        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] > 2:
            self.bots[bot_index].state[1] -= 0.3
        elif self.bots[bot_index].state[1] - self.bots[bot_index].target_state[1] < -2:
            self.bots[bot_index].state[1] += 0.3

        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] > 2:
            self.bots[bot_index].state[2] -= 0.3
        elif self.bots[bot_index].state[2] - self.bots[bot_index].target_state[2] < -2:
            self.bots[bot_index].state[2] += 0.3


    def get_drone_des_grid(self, drone_index, discrete_action):

        if discrete_action == 0: #action=0, x += 1.0
            self.quadrotors[drone_index].state[0] += self.grid_res
            self.quadrotors[drone_index].state[0] = np.clip(self.quadrotors[drone_index].state[0], -self.x_lim,  self.x_lim)
        elif discrete_action == 1: #action=1, x -= 1.0
            self.quadrotors[drone_index].state[0] -= self.grid_res
            self.quadrotors[drone_index].state[0] = np.clip(self.quadrotors[drone_index].state[0], -self.x_lim,  self.x_lim)
        elif discrete_action == 2: #action=2, y += 1.0
            self.quadrotors[drone_index].state[1] += self.grid_res
            self.quadrotors[drone_index].state[1] = np.clip(self.quadrotors[drone_index].state[1], -self.y_lim,  self.y_lim)
        elif discrete_action == 3: #action=3, y -= 1.0
            self.quadrotors[drone_index].state[1] -= self.grid_res
            self.quadrotors[drone_index].state[1] = np.clip(self.quadrotors[drone_index].state[1], -self.y_lim,  self.y_lim)
        elif discrete_action == 4: #action=4, z += 1.0
            self.quadrotors[drone_index].state[2] += self.grid_res
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  -1)
        elif discrete_action == 5: #action=5, z -= 1.0
            self.quadrotors[drone_index].state[2] -= self.grid_res
            self.quadrotors[drone_index].state[2] = np.clip(self.quadrotors[drone_index].state[2], -self.z_lim,  -1)
        else:
            print ("Invalid discrete action!")

        drone_current_state = np.copy(self.quadrotors[drone_index].state)
        return drone_current_state


    def get_closest_n_grids(self, current_pos, n):
        differences = current_pos-self.uncertainty_grids
        distances = np.sum(differences*differences,axis=1)
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        
        return sorted_indices[0:n]

    def get_closest_grid(self, current_pos):
        differences = current_pos-self.uncertainty_grids
        distances = np.sum(differences*differences,axis=1)
        min_ind = np.argmin(distances)
        
        return min_ind


    def visualize(self, agent_pos_dict=None, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim,
                                   self.x_lim, -self.y_lim, self.y_lim)
            fname = path.join(path.dirname(__file__), "assets/black.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane2.png")

            self.drone_transforms = []
            self.drones = []

            self.prey_transforms = []
            self.preys = []

            for i in range(self.n_agents):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(fname, 8., 8.))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(self.n_bots):
                self.prey_transforms.append(rendering.Transform())
                self.preys.append(rendering.Image(fname2, 8., 8.))
                self.preys[i].add_attr(self.prey_transforms[i])


        for i in range(self.n_bots):
            if self.bots[i].is_alive:
                self.viewer.add_onetime(self.preys[i])
                self.prey_transforms[i].set_translation(self.bots[i].state[0], self.bots[i].state[1])
                self.prey_transforms[i].set_rotation(self.bots[i].psid)
        
        for i in range(self.n_agents):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(self.quadrotors[i].state[0], self.quadrotors[i].state[1])
            self.drone_transforms[i].set_rotation(self.quadrotors[i].psi)
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
