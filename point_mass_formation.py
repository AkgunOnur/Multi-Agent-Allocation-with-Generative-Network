import gym
from gym import spaces, error, utils
from gym.utils import seeding
# from gym.envs.classic_control import rendering
# import rendering
import numpy as np
# import configparser
from os import path
# import itertools
# import random
# import pdb
from agents_dynamics import Agent
# from numpy.random import uniform
# from time import sleep
# from collections import deque
import warnings
# from PIL import Image
from dstar import *
import time

class AgentFormation(gym.Env):
    def __init__(self, visualization=True):
        np.set_printoptions(precision=4)
        warnings.filterwarnings('ignore')
        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 8
        self.n_agents = None
        self.visualization = visualization

        self.agents = []
        self.viewer = None

        # intitialize grid information
        self.map_lim = 20
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = self.map_lim  # width and height for uncertainty matrix


        # X, Y = np.mgrid[-self.map_lim : self.map_lim + 0.1 : 2*self.grid_res,
        #                 -self.map_lim : self.map_lim + 0.1 : 2*self.grid_res]
        X, Y = np.mgrid[0: self.map_lim: self.grid_res,
                        0: self.map_lim: self.grid_res]
        self.map_grids = np.vstack((X.flatten(), Y.flatten())).T

        self.N_prize = None
        self.agents_action_list = []
        self.prize_map = None

    def step(self, n_agents):
        #time.sleep(0.3)
        # print("infeasible_prizes: ", self.infeasible_prizes)
        # print("prize_exists: ", self.prize_exists)
        # print("N prize: ", self.N_prize)
        # print("------------------------")
        done = False
        total_reward = 0
        N_iteration = 100
        self.n_agents = n_agents
        self.assigned_agents_to_prizes = {i: [] for i in range(self.N_prize)}

        # Initialization points of agents
        self.init_list = []

        for x in np.arange(1, 5, self.grid_res):
            for y in np.arange(1, 5, self.grid_res):
                self.init_list.append([x, y]) 
        
        # Initialization of agents
        self.agents_locations = []
        agent_ind = 0
        while (agent_ind != self.n_agents):
            ind = np.random.choice(len(self.init_list))
            if self.init_list[ind] not in (self.obstacle_locations and  self.agents_locations and self.prize_locations):
                self.agents_locations.append(self.init_list[ind])
                self.agents.append(Agent(self.init_list[ind]))
                agent_ind += 1
            #print("agent initalizedL: ", agent_ind)
        #print("self.agents_locations: ", self.agents_locations)
        #print("map: ", self.observation)

        if self.visualization:
            self.visualize()
        
        # Initialization of trajectories
        self.agents_action_list = [[]*i for i in range(self.n_agents)]
        #print("self.n_agents: ", self.n_agents)
        for agent_ind in range(self.n_agents):
            feasible = False
            #print("Mark1")
            while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                #print("infeasible_prizes: ", self.infeasible_prizes)
                #print("feasioble: ", feasible)
                self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
            if np.sum(self.infeasible_prizes) == self.N_prize:
                total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                # print("infeasible_prizes: ", self.infeasible_prizes)
                # print("prize_exists: ", self.prize_exists)
                # print("N prize: ", self.N_prize)
                # print("------------------------")
                return total_reward, done, self.get_observation()
            ##print("agent initalizedL: ", agent_ind)
        #print("mark 2: ")
        for iteration in range(1, N_iteration + 1):
            for agent_ind in range(self.n_agents):

                if len(self.agents_action_list[agent_ind]) == 0:
                    feasible = False
                    while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                        self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
                    if np.sum(self.infeasible_prizes) == self.N_prize:
                        total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                        # print("infeasible_prizes: ", self.infeasible_prizes)
                        # print("prize_exists: ", self.prize_exists)
                        # print("N prize: ", self.N_prize)
                        # print("------------------------")
                        return total_reward, done, self.get_observation()

                # #print("agent_ind: ", agent_ind)
                # #print("self.agents_action_list: ", self.agents_action_list)
                current_action = (self.agents_action_list[agent_ind][0])

                total_reward -= 0.25
                prev_pos, current_pos = self.get_agent_desired_loc(agent_ind, current_action)


                self.agents_locations[agent_ind] = list(current_pos)

                if self.check_collision(current_pos):
                    print ("Agent {} has collided with the obstacle!".format(agent_ind))
                    # self.ds_map.get_map(self.prize_locations, self.agents_locations)
                    self.agents[agent_ind].state = np.copy(prev_pos)
                    continue

                del(self.agents_action_list[agent_ind][0])

                taken_prize_ind = next((index for index, prize in enumerate(self.prize_locations) if list(prize) == list(self.agents_locations[agent_ind])), -1)

                if taken_prize_ind >= 0:
                    self.prize_exists[taken_prize_ind] = False
                    self.infeasible_prizes[taken_prize_ind] = True
                    total_reward += 10.0

                    #Collected all prizes
                    if np.sum(self.prize_exists) == 0:
                        done = True
                        total_reward = total_reward + np.abs(total_reward) * (1 - iteration / N_iteration)
                        # print("infeasible_prizes: ", self.infeasible_prizes)
                        # print("prize_exists: ", self.prize_exists)
                        # print("N prize: ", self.N_prize)
                        # print("------------------------")
                        return total_reward, done, self.get_observation()

                    agents_for_prize = np.copy(self.assigned_agents_to_prizes[taken_prize_ind])
                    self.assigned_agents_to_prizes[taken_prize_ind] = []
                    for ind in agents_for_prize:
                        feasible = False
                        while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                            self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
                        if np.sum(self.infeasible_prizes) == self.N_prize:
                            total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                            # print("infeasible_prizes: ", self.infeasible_prizes)
                            # print("prize_exists: ", self.prize_exists)
                            # print("N prize: ", self.N_prize)
                            # print("------------------------")
                            return total_reward, done, self.get_observation()


                if self.visualization:
                    self.visualize()
                
                # self.ds_map.get_map(self.prize_locations, self.agents_locations)                
                

        total_reward = total_reward - np.sum(self.prize_exists) * 10.0
        #print("REWARD: ", total_reward)
        return total_reward, done, self.get_observation()

    def get_observation(self):
        for i in range(self.N_prize):
            if self.prize_exists[i]==False:
                    self.prize_map[self.prize_locations[i][0],self.prize_locations[i][1]] = 0

        self.observation[0,:,:] = np.copy(self.prize_map)
        self.observation[1,:,:] = np.copy(self.obstacle_map)
        # print("infeasible_prizes: ", self.infeasible_prizes)
        # print("prize_exists: ", self.prize_exists)
        # print("N prize: ", self.N_prize)
        # print("------------------------")

        return self.observation

    def reset(self, ds_map, obstacle_map, prize_map, agent_obs, map_lim, obs_y_list, obs_x_list):
        self.agents = []
        self.obs_y_list = obs_y_list
        self.obs_x_list = obs_x_list
        self.map_lim = map_lim

        self.ds_map = ds_map

        #P = np.where(prize_map==1)
        self.N_prize = len(prize_map)
        self.prize_exists = np.ones(self.N_prize, dtype=bool)
        self.infeasible_prizes = np.zeros(self.N_prize, dtype=bool)
        self.prize_map = agent_obs[0,:,:]
        self.prize_locations = []

        #O = np.where(obstacle_map==1)
        self.N_obstacle = len(obstacle_map)
        self.obstacle_map = agent_obs[1,:,:]
        self.obstacle_locations = []
        
        self.observation = agent_obs
        
        #get prize locations
        for i in range(self.N_prize):
            self.prize_locations.append([prize_map[i][0],prize_map[i][1]])

        #get obstacle locations
        for n in range(self.N_obstacle):
            self.obstacle_locations.append([obstacle_map[n][0],obstacle_map[n][1]])

        return self.observation

    def get_obstacle_locations(self):
        ds_map = Map(self.map_lim, self.map_lim)
        ds_map.set_obstacle([(i, j) for i, j in zip(self.obs_y_list, self.obs_x_list)])
        return ds_map, self.obstacle_map
        
    def create_trajectory(self, agent_ind):
        action_list, pos_list = [], []
        self.ds_map, self.obstacle_locations = self.get_obstacle_locations()
        self.dstar = Dstar(self.ds_map)
        # probabilistic way
        # euclidean_dist = np.sum((self.agents[agent_ind].state - self.prize_locations)**2, axis=1) 
        # euclidean_dist[~self.prize_exists] = 0.0
        # sum_euclidean_dist = np.sum(euclidean_dist)
        # prob_values_pr = euclidean_dist / sum_euclidean_dist
        # target_cnt = np.array([len(self.assigned_agents_to_prizes[element]) for element in self.assigned_agents_to_prizes]) + 1
        # euclidean_dist = euclidean_dist * target_cnt * 10
        # euclidean_dist[~self.prize_exists] = 0.0
        # sum_euclidean_dist = np.sum(euclidean_dist)
        # prob_values = euclidean_dist / sum_euclidean_dist
        # target_prize = np.random.choice(np.arange(self.N_prize), p=prob_values)
        
        # euclidean_dist_pr = np.sum((self.agents[agent_ind].state - self.prize_locations)**2, axis=1) 
        # euclidean_dist_pr[~self.prize_exists] = 1e9
        euclidean_dist = np.sum((self.agents[agent_ind].state - self.prize_locations)**2, axis=1) 
        target_cnt = np.array([len(self.assigned_agents_to_prizes[element]) for element in self.assigned_agents_to_prizes]) + 1
        ##print("$")
        euclidean_dist = euclidean_dist * target_cnt * 10
        # euclidean_dist[~self.prize_exists] = 1e9
        euclidean_dist[self.infeasible_prizes] = 1e9
        target_prize = np.argmin(euclidean_dist)
        self.assigned_agents_to_prizes[target_prize].append(agent_ind)

        start = self.ds_map.map[int(self.agents_locations[agent_ind][0])][int(self.agents_locations[agent_ind][1])]
        end = self.ds_map.map[int(self.prize_locations[target_prize][0])][int(self.prize_locations[target_prize][1])]

        # print("start: ", int(self.agents_locations[agent_ind][0]), int(self.agents_locations[agent_ind][1]))
        # print("end: ", int(self.prize_locations[target_prize][0]), int(self.prize_locations[target_prize][1]))
        # self.ds_map.get_map(self.prize_locations)
        feasible, pos_list, action_list = self.dstar.run(start, end)
        #print("pos_list: ", pos_list)
        if feasible == True:
            feasible = self.check_feasibility(pos_list)
            if feasible == False:
                self.infeasible_prizes[target_prize] = True # prize is not accessible

        elif feasible == False:
            self.infeasible_prizes[target_prize] = True # prize is not accessible

        return action_list, pos_list, feasible

    def check_feasibility(self, pos_list):
        for pos in pos_list:
            if list(pos) in self.obstacle_locations:
                return False
        return True

    def check_collision(self, agent_pos):
        if list(agent_pos) in self.obstacle_locations:
            return True
        return False

    def get_agent_desired_loc(self, agent_index, discrete_action):
        agent_prev_state = np.copy(self.agents[agent_index].state)

        if discrete_action == 0:  # action=0, x += 1.0
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0] + self.grid_res, 0, self.map_lim)
        elif discrete_action == 1:  # action=1, x -= 1.0
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0] - self.grid_res, 0, self.map_lim)
        elif discrete_action == 2:  # action=2, y += 1.0
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] + self.grid_res, 0, self.map_lim)
        elif discrete_action == 3:  # action=3, y -= 1.0
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] - self.grid_res, 0, self.map_lim)
        elif discrete_action == 4:  # action=4, x += 1.0, y += 1.0
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0] + self.grid_res, 0, self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] + self.grid_res, 0, self.map_lim)
        elif discrete_action == 5:  # action=5, x += 1.0, y -= 1.0
            self.agents[agent_index].state[0]  = np.clip(self.agents[agent_index].state[0] + self.grid_res, 0, self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] - self.grid_res, 0, self.map_lim)
        elif discrete_action == 6:  # action=6, x -= 1.0, y += 1.0
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0] - self.grid_res, 0, self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] + self.grid_res, 0, self.map_lim)
        elif discrete_action == 7:  # action=7, x -= 1.0, y -= 1.0
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0] - self.grid_res, 0, self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1] - self.grid_res, 0, self.map_lim)
        elif discrete_action == -1:  # action=-1 stop
            print("No action executed!")
        else:
            print("Invalid discrete action!")

        agent_current_state = np.copy(self.agents[agent_index].state)

        return agent_prev_state, agent_current_state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None