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
        done = False
        total_reward = 0
        N_iteration = 100
        self.n_agents = n_agents
        self.assigned_agents_to_prizes = {i: [] for i in range(self.N_prize)}

        # Initialization points of agents
        self.init_list = []

        for x in np.arange(1, 10, self.grid_res):
            for y in np.arange(1, 10, self.grid_res):
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

        if self.visualization:
            self.visualize()

        # === Onur'a step func sor ===

        # Initialization of trajectories
        self.agents_action_list = [[]*i for i in range(self.n_agents)]
        for agent_ind in range(self.n_agents):
            feasible = False
            while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
            if np.sum(self.infeasible_prizes) == self.N_prize:
                total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                return total_reward, done, self.get_observation()


        for iteration in range(1, N_iteration + 1):
            for agent_ind in range(self.n_agents):

                if len(self.agents_action_list[agent_ind]) == 0:
                    feasible = False
                    while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                        self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
                    if np.sum(self.infeasible_prizes) == self.N_prize:
                        total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                        return total_reward, done, self.get_observation()

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
                        return total_reward, done, self.get_observation()

                    agents_for_prize = np.copy(self.assigned_agents_to_prizes[taken_prize_ind])
                    self.assigned_agents_to_prizes[taken_prize_ind] = []
                    for ind in agents_for_prize:
                        feasible = False
                        while (feasible == False and np.sum(self.infeasible_prizes) < self.N_prize): # check if there are still accessible prizes
                            self.agents_action_list[agent_ind], pos_list, feasible = self.create_trajectory(agent_ind)
                        if np.sum(self.infeasible_prizes) == self.N_prize:
                            total_reward = total_reward - np.sum(self.prize_exists) * 10.0
                            return total_reward, done, self.get_observation()


                if self.visualization:
                    self.visualize()
                
                # self.ds_map.get_map(self.prize_locations, self.agents_locations)                
                

        total_reward = total_reward - np.sum(self.prize_exists) * 10.0

        return total_reward, done, self.get_observation()

    # def get_observation(self):
    #     neigbor_grids = [[0,1], [0, -1], [1,0], [-1, 0], [0,0]]
    #     # Vector based observations
    #     # state_array = np.array([self.agents[i].state for i in range(self.n_agents)])
    #     # state_obs = np.zeros((self.n_agents, self.n_agents*3))

    #     # # 1 + n_agents*3
    #     # #observation list = [battery_status,x11,x12,..x1n,y11,y12,..y1n,z11,z12,..z1n]
    #     # for axis in range(2): #x, y, z axes
    #     #     state_tile = np.tile(state_array.T[axis],(self.n_agents,1))
    #     #     state_tile_mask = np.copy(state_tile)
    #     #     np.fill_diagonal(state_tile_mask, 0)
    #     #     state_obs[:,axis*self.n_agents:(axis+1)*self.n_agents] = np.copy(state_tile.T - state_tile_mask) / self.map_lim

    #     # final_obs = np.c_[self.battery_status, state_obs]

    #     self.prize_map = np.zeros((self.out_shape, self.out_shape))
    #     self.obstacle_map = np.zeros((self.out_shape, self.out_shape))
    #     self.observation = np.zeros((2,self.out_shape, self.out_shape))

    #     for i in range(self.N_prize):
    #         if self.prize_exists[i]:
    #             for x_n, y_n in neigbor_grids:
    #                 x = np.clip(self.prize_locations[i][0] + x_n, 0, self.map_lim - 1)
    #                 y = np.clip(self.prize_locations[i][1] + y_n, 0, self.map_lim - 1)
    #                 self.prize_map[x,y] = 1

    #     for obs_point in self.obstacle_locations:
    #         x,y = obs_point[0], obs_point[1]
    #         self.obstacle_map[x][y] = 1


    #     self.observation[0,:,:] = np.copy(self.prize_map)
    #     self.observation[1,:,:] = np.copy(self.obstacle_map)
    #     self.prize_map = self.prize_map.reshape(1, self.out_shape, self.out_shape)
    #     return [self.prize_map, self.obstacle_map]

    def reset(self, ds_map, obstacle_map, prize_map, agent_obs):
        self.agents = []

        self.ds_map = ds_map

        P = np.where(prize_map==1)
        self.N_prize = len(P)
        self.prize_exists = np.ones(self.N_prize, dtype=bool)
        self.infeasible_prizes = np.zeros(self.N_prize, dtype=bool)
        self.prize_map = prize_map
        self.prize_locations = []

        O = np.where(obstacle_map==1)
        self.N_obstacle = len(O)
        self.obstacle_map = obstacle_map
        self.obstacle_locations = []
        
        self.observation = agent_obs
        
        #get prize locations
        for i in range(self.N_prize):
            self.prize_locations.append(list(P[i][0],P[i][1]))

        #get obstacle locations
        for n in range(self.N_obstacle):
            self.obstacle_locations.append(list(O[n][0],O[n][1]))

        return self.observation

    def create_trajectory(self, agent_ind):
        action_list, pos_list = [], []
        #self.ds_map, self.obstacle_locations = self.get_obstacle_locations()
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
        euclidean_dist = euclidean_dist * target_cnt * 10
        # euclidean_dist[~self.prize_exists] = 1e9
        euclidean_dist[self.infeasible_prizes] = 1e9
        target_prize = np.argmin(euclidean_dist)
        self.assigned_agents_to_prizes[target_prize].append(agent_ind)

        start = self.ds_map.map[int(self.agents_locations[agent_ind][0])][int(self.agents_locations[agent_ind][1])]
        end = self.ds_map.map[int(self.prize_locations[target_prize][0])][int(self.prize_locations[target_prize][1])]

        feasible, pos_list, action_list = self.dstar.run(start, end)
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

    # def get_obstacle_locations(self):
    #     obs_x_list = []
    #     obs_y_list = []
    #     obstacle_locations = []
    #     self.obstacle_list = np.array([])
    #     ds_map = Map(self.map_lim, self.map_lim)

    #     # self.generated_obstacles[0].append([0, self.map_lim, 10, 12])
    #     # self.generated_obstacles[0].append([11, 12, 0, 12])

    #     # Manual curriculum
    #     if self.curriculum_index == None:
    #         self.obstacle_list = np.copy(self.generated_obstacles[0])
    #     else:
    #         if self.level == "easy":
    #             self.obstacle_list = np.array([])
    #         elif self.level == "medium":
    #             self.obstacle_list = np.array(self.medium_obstacle_list[self.curriculum_index])
    #         elif self.level == "hard":
    #             self.obstacle_list = np.array(self.hard_obstacle_list[self.curriculum_index])
    #         elif self.level == "random":
    #             self.obstacle_list = np.array(self.random_obstacles_list[0])

    #     self.wall_list = np.array([[0, self.map_lim, 0, 1], [0, self.map_lim, self.map_lim-1, self.map_lim], 
    #                                [self.map_lim-1, self.map_lim, 0, self.map_lim], [0, 1, 0, self.map_lim]])


    #     for obs in self.wall_list:
    #         for x in range(obs[0], obs[1]):
    #             for y in range(obs[2], obs[3]):
    #                 current_pos = [y, x]
    #                 obs_x_list.append(x)
    #                 obs_y_list.append(y)

    #                 if current_pos not in obstacle_locations:
    #                     obstacle_locations.append(current_pos)
        
    #     for obs in self.obstacle_list:
    #         for x in range(obs[0], obs[1]):
    #             for y in range(obs[2], obs[3]):
    #                 current_pos = [y, x]
    #                 obs_x_list.append(x)
    #                 obs_y_list.append(y)

    #                 if current_pos not in obstacle_locations:
    #                     obstacle_locations.append(current_pos)

    #     ds_map.set_obstacle([(i, j) for i, j in zip(obs_x_list, obs_y_list)])
        
    #     return ds_map, obstacle_locations

    # def visualize(self, mode='human'):
    #     grids = []
    #     station_transform = []

    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(1000, 1000)
    #         self.viewer.set_bounds(0, self.map_lim - 1, 0, self.map_lim - 1)
    #         fname = path.join(path.dirname(__file__), "assets/drone.png")
    #         fname_prize = path.join(path.dirname(__file__), "assets/prize.jpg")

    #         background = rendering.make_polygon([(0, 0), (0, self.map_lim - 1),
    #                                              (self.map_lim - 1, self.map_lim - 1), 
    #                                              (self.map_lim - 1, 0)])

    #         background_transform = rendering.Transform()
    #         background.add_attr(background_transform)
    #         background.set_color(0., 0.9, 0.5)  # background color
    #         self.viewer.add_geom(background)


    #         wall_list = np.array([[0, 0.5, 0, self.map_lim - 1], 
    #                               [self.map_lim - 1.5, self.map_lim - 1, 0, self.map_lim - 1], 
    #                               [0, self.map_lim - 1, self.map_lim - 1.5, self.map_lim - 1], 
    #                               [0, self.map_lim - 1, 0, 0.5]])
    #         for i in range(wall_list.shape[0]):
    #             obstacle = rendering.make_polygon([(wall_list[i][0], wall_list[i][2]),  # xmin, ymin
    #                                                (wall_list[i][0], wall_list[i][3]),  # xmin, ymax
    #                                                (wall_list[i][1], wall_list[i][3]),  # xmax, ymax
    #                                                (wall_list[i][1], wall_list[i][2])])  # xmax, ymin

    #             obstacle_transform = rendering.Transform()
    #             obstacle.add_attr(obstacle_transform)
    #             obstacle.set_color(.8, .3, .3)  # obstacle color
    #             self.viewer.add_geom(obstacle)

            
    #         obs_list = self.obstacle_list - 0.5
    #         for i in range(obs_list.shape[0]):
    #             obstacle = rendering.make_polygon([(obs_list[i][0], self.map_lim - 1 - obs_list[i][3]),
    #                                                 (obs_list[i][0], self.map_lim - 1 - obs_list[i][2]),
    #                                                 (obs_list[i][1], self.map_lim - 1 - obs_list[i][2]),
    #                                                 (obs_list[i][1], self.map_lim - 1 - obs_list[i][3])])

    #             obstacle_transform = rendering.Transform()
    #             obstacle.add_attr(obstacle_transform)
    #             obstacle.set_color(.8, .3, .3) #obstacle color
    #             self.viewer.add_geom(obstacle)


    #         # self.grid_points = np.array([[0, 0.1, 0, 20], [1, 1.1, 0, 20], [2, 2.1, 0, 20], [3, 3.1, 0, 20], [0, 20, 0, 0.1], [0, 20, 1, 1.1]])
    #         for j in range(len(self.grid_points)):
    #             grid = rendering.make_polygon([(self.grid_points[j][0], self.grid_points[j][2]),
    #                                            (self.grid_points[j][0], self.grid_points[j][3]),
    #                                            (self.grid_points[j][1], self.grid_points[j][3]),
    #                                            (self.grid_points[j][1], self.grid_points[j][2])])

    #             grid_transform = rendering.Transform()
    #             grid.add_attr(grid_transform)
    #             grid.set_color(0., 0.65, 1.0)  # grid color
    #             self.viewer.add_geom(grid)

    #         self.agent_transforms = []
    #         self.agents_img = []
    #         self.prizes = []
    #         self.prize_transformations = []

    #         for i in range(10):
    #             self.agent_transforms.append(rendering.Transform())
    #             self.agents_img.append(
    #                 rendering.Image(fname, 1, 1))  # agent size
    #             self.agents_img[i].add_attr(self.agent_transforms[i])

    #         for i in range(10):
    #             self.prize_transformations.append(rendering.Transform())
    #             self.prizes.append(rendering.Image(
    #                 fname_prize, 1., 1.))  # prize size
    #             self.prizes[i].add_attr(self.prize_transformations[i])

    #     for i in range(self.n_agents):
    #         self.viewer.add_onetime(self.agents_img[i])
    #         self.agent_transforms[i].set_translation(self.agents[i].state[1], self.map_lim - 1 - self.agents[i].state[0])
    #         self.agent_transforms[i].set_rotation(0)

    #     for i in range(self.N_prize):
    #         if self.prize_exists[i] == True:
    #             self.viewer.add_onetime(self.prizes[i])
    #             self.prize_transformations[i].set_translation(self.prize_locations[i][1], self.map_lim - 1 - self.prize_locations[i][0])
    #             self.prize_transformations[i].set_rotation(0)

    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None