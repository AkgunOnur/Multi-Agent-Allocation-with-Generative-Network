import gym
from gym import spaces, error, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
# import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from numpy.random import uniform
from time import sleep
from collections import deque
import warnings
import time
from PIL import Image

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

class Agent:
    def __init__(self, state0):
        self.state = np.array(state0)

class AgentFormation(gym.Env):
    def __init__(self, generated_map, max_steps=5000, visualization=False):
        super().__init__()
        np.set_printoptions(precision=4)
        warnings.filterwarnings('ignore')
        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 8
        self.dim_actions = 1
        self.n_agents = 1
        self.visualization = visualization
        self.gen_map = generated_map

        self.viewer = None

        # intitialize grid information
        self.map_lim = 10
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = self.map_lim  # width and height for uncertainty matrix
        self.N_prize = None
        self.agent_locations = [1,1]
        self.agents = None
        
        
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(3, self.out_shape, self.out_shape), dtype=np.uint8)
        self.current_steps = 0
        self.max_steps = max_steps

        self.predefined_obtacles = self.get_indices([i  for i in range(self.map_lim**2) if i% self.map_lim == 0 or i % self.map_lim == self.map_lim - 1 \
                                                                                        or i // self.map_lim == 0 or i // self.map_lim == self.map_lim - 1])
        self.obstacle_locations = None
        self.prize_locations = None

            

    def get_indices(self, numbers):
        index = []
        for number in numbers:
            x = number // 10
            y = number % 10
            index.append([x,y])
        return np.array(index)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reward = -0.1
        info = dict()

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            # Update the info dict to signal that the limit was exceeded
            info['time_limit_reached'] = True

        self.get_agent_desired_loc(action)
        for i, prize_loc in enumerate(self.prize_locations):
            if np.array_equal(self.agents[0].state, prize_loc) and self.prize_exists[i] == True:
                self.prize_exists[i] = False
                reward = 10.0
                if np.sum(self.prize_exists) == 0:
                    done = True
        
        if self.visualization:
            self.visualize()  


        return self.get_observation(), reward, done, info

    def get_observation(self):
        self.observation_map = np.zeros((3,self.out_shape, self.out_shape))
        self.neighbor_grids = np.array([[0,0],[-1,0],[1,0],[0,1],[0,-1]])

        for i in range(self.n_agents):
            for neighbor_grid in self.neighbor_grids:
                current_grid = np.clip(self.agents[i].state + neighbor_grid, 0, self.map_lim)
                self.observation_map[0, current_grid[0], current_grid[1]] = 1

        
        for obs_point in self.obstacle_locations:
            x,y = obs_point[0], obs_point[1]
            self.observation_map[1,x,y] = 1

        for i, prize_loc in enumerate(self.prize_locations):
            if self.prize_exists[i] == True:
                x,y = prize_loc[0], prize_loc[1]
                self.observation_map[2,x,y] = 1


        # print ("agent_loc: ", self.agents[0].state)
        # print (self.observation_map[0, :, :])
        # print ("\n")
        # print (self.observation_map[1, :, :])
        # print ("\n")
        # print ("prize_exists: ", self.prize_exists)
        # print (self.observation_map[2, :, :])
        # print ("\n")
        

        
        # for r in range(self.map_lim):
        #     for c in range(self.map_lim):
        #         if [r,c] not in self.prize_locations and [r,c] not in self.obstacle_locations:
        #             self.free_area[r,c] = 1

        # for obs_point in self.obstacle_locations:
        #     x,y = obs_point[0], obs_point[1]
        #     self.obstacle_map[x][y] = 1
        # self.prize_map = self.prize_map.reshape(1, self.out_shape, self.out_shape)

        # for prize_loc in self.prize_locations:
        #     self.observation_map[prize_loc[0],prize_loc[1]] = 1

        # for agent_ind in range(self.n_agents):
        #     self.observation_map[int(self.agents[agent_ind].state[0]),
        #                          int(self.agents[agent_ind].state[1])] = 1

        
        return self.observation_map


    def reset(self):
        self.current_step = 0
        self.grid_points = []
        self.obstacle_locations = []
        # self.agent_locations = []
        # self.prize_locations = []
        # self.agents = []
        
        self.agents = [Agent(self.agent_locations)]

        # Points of grids to be drawn
        x_list = np.arange(0, self.map_lim, self.grid_res)
        y_list = np.arange(0, self.map_lim, self.grid_res)
        eps = 0.15
        for x in x_list:
            grid = [x+0.5, x+0.5+eps, 0, self.map_lim]
            self.grid_points.append(grid)

        for y in y_list:
            grid = [0, self.map_lim, y+0.5, y+0.5+eps]
            self.grid_points.append(grid)
        
        # Initialization points of agents
        x_list = np.arange(1, 3, self.grid_res)
        y_list = np.arange(1, 3, self.grid_res)
        self.init_list = []

        for x in x_list:
            for y in y_list:
                grid0 = [x, y]
                self.init_list.append(grid0)     

        # agent_ind = 0
        # while (agent_ind != self.n_agents):
        #     ind = np.random.choice(len(self.init_list))
        #     if self.init_list[ind] not in self.obstacle_locations and self.init_list[ind] not in self.agent_locations:
        #         self.agent_locations.append(self.init_list[ind])
        #         self.agents.append(Agent(self.init_list[ind]))
        #         agent_ind += 1

        # prize_ind = 0
        # while (prize_ind != self.N_prize):
        #     loc = list([np.random.choice(range(5, 7)), np.random.choice(range(8, self.map_lim - 1))])
        #     loc = [8,1]
        #     if loc not in self.obstacle_locations and loc not in self.prize_locations and loc not in self.init_list:
        #         self.prize_locations.append(loc)
        #         prize_ind += 1

        self.obstacle_locations, self.prize_locations = self.get_obstacle_locations()
        self.N_prize = len(self.prize_locations)
        self.prize_exists = np.ones(self.N_prize, dtype=bool) 

        return self.get_observation()

    def check_collision(self, agent_pos):
        if list(agent_pos) in self.obstacle_locations:
            return True

        return False

    def get_obstacle_locations(self):
        obstacle_loc = []
        prize_loc = []

        self.gen_map[self.predefined_obtacles[:,0], self.predefined_obtacles[:,1]] = 1 # predefined walls
        self.gen_map[1,1] = 0 # Agent start location, obstacle free
        # print (self.gen_map)
        
        for i in range(self.map_lim):
            for j in range(self.map_lim):
                if self.gen_map[i][j] == 1:
                    obstacle_loc.append([i,j])
                elif self.gen_map[i][j] == 2:
                    prize_loc.append([i,j])

        return obstacle_loc, prize_loc

         

    def get_agent_desired_loc(self, discrete_action, agent_index = 0):
        agent_prev_state = np.copy(self.agents[agent_index].state)

        if discrete_action == 0:  # action=0, x += 1.0
            self.agents[agent_index].state[0] += self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
        elif discrete_action == 1:  # action=1, x -= 1.0
            self.agents[agent_index].state[0] -= self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
        elif discrete_action == 2:  # action=2, y += 1.0
            self.agents[agent_index].state[1] += self.grid_res
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == 3:  # action=3, y -= 1.0
            self.agents[agent_index].state[1] -= self.grid_res
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == 4:  # action=4, x += 1.0, y += 1.0
            self.agents[agent_index].state[0] += self.grid_res
            self.agents[agent_index].state[1] += self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == 5:  # action=5, x += 1.0, y -= 1.0
            self.agents[agent_index].state[0] += self.grid_res
            self.agents[agent_index].state[1] -= self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == 6:  # action=6, x -= 1.0, y += 1.0
            self.agents[agent_index].state[0] -= self.grid_res
            self.agents[agent_index].state[1] += self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == 7:  # action=7, x -= 1.0, y -= 1.0
            self.agents[agent_index].state[0] -= self.grid_res
            self.agents[agent_index].state[1] -= self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], 0,  self.map_lim)
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], 0,  self.map_lim)
        elif discrete_action == -1:  # action=-1 stop
            print("No action executed!")
        else:
            print("Invalid discrete action!")

        agent_current_state = np.copy(self.agents[agent_index].state)

        if self.check_collision(self.agents[agent_index].state):
            self.agents[agent_index].state = np.copy(agent_prev_state)
            # print ("Collision detected!")
            # time.sleep(0.1)


        # return agent_current_state

    # def get_obstacle_locations(self):
    #     obs_x_list = []
    #     obs_y_list = []
    #     obstacle_locations = []

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
        
        
    #     return obstacle_locations

    def visualize(self, mode='human'):
        grids = []
        station_transform = []

        if self.viewer is None:
            self.viewer = rendering.Viewer(200, 200)
            self.viewer.set_bounds(0, self.map_lim - 1, 0, self.map_lim - 1)
            fname = path.join(path.dirname(__file__), "assets/drone.png")
            fname_prize = path.join(path.dirname(__file__), "assets/prize.jpg")

            background = rendering.make_polygon([(0, 0), (0, self.map_lim - 1),
                                                 (self.map_lim - 1, self.map_lim - 1), 
                                                 (self.map_lim - 1, 0)])

            background_transform = rendering.Transform()
            background.add_attr(background_transform)
            background.set_color(0., 0.9, 0.5)  # background color
            self.viewer.add_geom(background)


            wall_list = np.array([[0, 0.5, 0, self.map_lim - 1], 
                                  [self.map_lim - 1.5, self.map_lim - 1, 0, self.map_lim - 1], 
                                  [0, self.map_lim - 1, self.map_lim - 1.5, self.map_lim - 1], 
                                  [0, self.map_lim - 1, 0, 0.5]])
            for i in range(wall_list.shape[0]):
                obstacle = rendering.make_polygon([(wall_list[i][0], wall_list[i][2]),  # xmin, ymin
                                                   (wall_list[i][0], wall_list[i][3]),  # xmin, ymax
                                                   (wall_list[i][1], wall_list[i][3]),  # xmax, ymax
                                                   (wall_list[i][1], wall_list[i][2])])  # xmax, ymin

                obstacle_transform = rendering.Transform()
                obstacle.add_attr(obstacle_transform)
                obstacle.set_color(.8, .3, .3)  # obstacle color
                self.viewer.add_geom(obstacle)

            
            for obs_loc in self.obstacle_locations:
                obs_0 = np.array(obs_loc) - 0.5
                obs_1 = np.array(obs_loc) + 0.5
                obstacle = rendering.make_polygon([(obs_0[1], self.map_lim - 1 - obs_0[0]), 
                                                   (obs_0[1], self.map_lim - 1 - obs_1[0]), 
                                                   (obs_1[1], self.map_lim - 1 - obs_1[0]), 
                                                   (obs_1[1], self.map_lim - 1 - obs_0[0])])

                # obstacle = rendering.make_polygon([(obs_list[i][0], self.map_lim - 1 - obs_list[i][3]), 
                #                                     (obs_list[i][0], self.map_lim - 1 - obs_list[i][2]), 
                #                                     (obs_list[i][1], self.map_lim - 1 - obs_list[i][2]), 
                #                                     (obs_list[i][1], self.map_lim - 1 - obs_list[i][3])])

                obstacle_transform = rendering.Transform()
                obstacle.add_attr(obstacle_transform)
                obstacle.set_color(.8, .3, .3) #obstacle color
                self.viewer.add_geom(obstacle)


            # self.grid_points = np.array([[0, 0.1, 0, 20], [1, 1.1, 0, 20], [2, 2.1, 0, 20], [3, 3.1, 0, 20], [0, 20, 0, 0.1], [0, 20, 1, 1.1]])
            for j in range(len(self.grid_points)):
                grid = rendering.make_polygon([(self.grid_points[j][0], self.grid_points[j][2]),
                                               (self.grid_points[j][0], self.grid_points[j][3]),
                                               (self.grid_points[j][1], self.grid_points[j][3]),
                                               (self.grid_points[j][1], self.grid_points[j][2])])

                grid_transform = rendering.Transform()
                grid.add_attr(grid_transform)
                grid.set_color(0., 0.65, 1.0)  # grid color
                self.viewer.add_geom(grid)

            self.agent_transforms = []
            self.agents_img = []
            self.prizes = []
            self.prize_transformations = []

            for i in range(1):
                self.agent_transforms.append(rendering.Transform())
                self.agents_img.append(
                    rendering.Image(fname, 1, 1))  # agent size
                self.agents_img[i].add_attr(self.agent_transforms[i])

            for i in range(self.N_prize):
                self.prize_transformations.append(rendering.Transform())
                self.prizes.append(rendering.Image(
                    fname_prize, 1., 1.))  # prize size
                self.prizes[i].add_attr(self.prize_transformations[i])

        for i in range(self.n_agents):
            self.viewer.add_onetime(self.agents_img[i])
            self.agent_transforms[i].set_translation(self.agents[i].state[1], self.map_lim - 1 - self.agents[i].state[0])
            self.agent_transforms[i].set_rotation(0)

        for i in range(self.N_prize):
            if self.prize_exists[i] == True:
                self.viewer.add_onetime(self.prizes[i])
                self.prize_transformations[i].set_translation(self.prize_locations[i][1], self.map_lim - 1 - self.prize_locations[i][0])
                self.prize_transformations[i].set_rotation(0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
