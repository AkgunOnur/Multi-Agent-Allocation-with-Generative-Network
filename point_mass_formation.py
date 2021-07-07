import gym
from gym import spaces, error, utils
from gym.utils import seeding
# from gym.envs.classic_control import rendering
import numpy as np
import configparser
from os import path
import itertools
import random
import pdb
from agent_dynamics import Agent
from numpy.random import uniform
from time import sleep
from collections import deque
import warnings
import time

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, action=None):
        self.parent = parent
        self.position = position
        self.action = action

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        for pos1, pos2 in zip(self.position, other.position):
            if pos1 != pos2:
                return False
        return True


class AgentFormation(gym.Env):
    def __init__(self, visualization=True):
        warnings.filterwarnings('ignore')
        # number of actions per agent which are desired positions and yaw angle
        self.n_action = 4
        self.observation_dim = 4
        self.dim_actions = 1
        self.n_agents = None
        self.visualization = visualization
        self.action_dict = {0:"Xp", 1:"Xn", 2:"Yp", 3:"Yn"}

        state0 = [0., 0.]
        self.agents = []
        self.viewer = None
        self.dtau = 1e-3
        self.agent_status = None
        self.agent_is_stuck = None

        self.action_space = spaces.Discrete(self.n_action)

        # intitialize grid information
        self.map_lim = 20
        self.grid_res = 1.0  # resolution for grids
        self.out_shape = 21  # width and height for uncertainty matrix
        self.dist = 5.0  # distance threshold
        self.N_closest_grid = 1
        self.neighbour_grids = 8
        self.agent_pos_index = None
        self.safest_indices = None
        self.grid_points = None

        self.obstacle_start = np.array([[-20, -20], [-20, -20], [20, -20], [-20, 20]]) 
        self.obstacle_end = np.array([[-19, 20.1], [20.1, -19], [20.1, 20.1], [20.1, 20.1]])
        
        self.obstacle_indices = None
        self.obstacle_pos_xy = None

        X, Y = np.mgrid[-self.map_lim : self.map_lim + 0.1 : 2*self.grid_res, 
                        -self.map_lim : self.map_lim + 0.1 : 2*self.grid_res]
        self.map_grids = np.vstack((X.flatten(), Y.flatten())).T

        self.N_prize = None
        self.pos_prize_loc = None
        self.agents_action_list = []
        self.prize_map = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, n_agents):
        done = False
        total_reward = 0
        N_iteration = 100
        self.n_agents = n_agents  
        self.agent_status = np.ones(self.n_agents)
        self.agent_pos_index = -1 * np.ones(self.n_agents)
        self.agent_is_stuck = np.zeros(self.n_agents)
        self.assigned_agents_to_prizes = {i:[] for i in range(self.N_prize)} 

        # Initialization of agents
        state0_list = np.random.choice(range(self.n_agents),(self.n_agents,),replace=False)
        for agent_ind in range(0, self.n_agents):
            state0 = self.init_pos_list[state0_list[agent_ind]]
            self.agents.append(Agent(state0))
            self.agent_pos_index[agent_ind] = self.get_closest_grid(state0)

        # closest grids to prizes are 1, others are 0
        self.prize_map = np.reshape(self.prize_map, (self.out_shape, self.out_shape))
        self.agents_action_list = [[]*i for i in range(self.n_agents)]
        
        for agent_ind in range(self.n_agents):
            self.agents_action_list[agent_ind] = self.create_trajectory(agent_ind)

        # print ("\n")

        for iter in range(1, N_iteration + 1):
            for agent_ind in range(self.n_agents):
                
                if self.agent_status[agent_ind] == 0:
                    # print ("Agent {0} failed, it can't fly any longer!".format(agent_ind+1))
                    continue

                if np.any(self.agents_action_list[agent_ind]) == False:
                    self.agents_action_list[agent_ind] = self.create_trajectory(agent_ind)
                    # print ("assigned_agents_to_prizes: ", self.assigned_agents_to_prizes)
                    
                current_action = (self.agents_action_list[agent_ind][0])
                del(self.agents_action_list[agent_ind][0])

                total_reward -= 1.0
                prev_pos, current_pos = self.get_agent_desired_grid(agent_ind, current_action)
                current_grid = self.get_closest_grid(current_pos)
                selected_indices = np.setdiff1d(range(self.n_agents), agent_ind)

                if current_grid in self.agent_pos_index[selected_indices]:
                    # print ("Non available grid for agent {}!".format(agent_ind+1))
                    self.agents[agent_ind].state = np.copy(prev_pos)
                    continue
                
                if self.check_collision(current_grid): 
                    # print ("Agent {} has collided with the obstacle!".format(agent_ind+1))
                    self.agents[agent_ind].state = np.copy(prev_pos)
                    continue

                self.agent_pos_index[agent_ind] = self.get_closest_grid(current_pos)

                
                if np.any(self.agent_pos_index[agent_ind] == self.prize_grids):
                    prize_index = np.where(self.agent_pos_index[agent_ind] == self.prize_grids)[0][0]
                    self.prize_exists[prize_index] = False
                    total_reward += 10.0

                    if np.sum(self.prize_exists) == 0:
                        done = True
                        total_reward = total_reward + np.abs(total_reward) * (1 - iter / N_iteration) 
                        return total_reward, done, self.get_observation()
                    
                    # print ("self.assigned_agents_to_prizes: ", self.assigned_agents_to_prizes)
                    # print ("prize_index: ", prize_index)
                    agents_for_prize = np.copy(self.assigned_agents_to_prizes[prize_index])
                    self.assigned_agents_to_prizes[prize_index] = []
                    for ind in agents_for_prize:
                        self.agents_action_list[ind] = self.create_trajectory(ind)

                    # print ("assigned_agents_to_prizes: ", self.assigned_agents_to_prizes)
                                        

                if self.visualization:
                    self.visualize()


        total_reward = total_reward - np.sum(self.prize_exists) * 10.0

        return total_reward, done, self.get_observation()

        

    def get_observation(self):

        #Vector based observations
        # state_array = np.array([self.agents[i].state for i in range(self.n_agents)])
        # state_obs = np.zeros((self.n_agents, self.n_agents*3))

        # # 1 + n_agents*3 
        # #observation list = [battery_status,x11,x12,..x1n,y11,y12,..y1n,z11,z12,..z1n]
        # for axis in range(2): #x, y, z axes
        #     state_tile = np.tile(state_array.T[axis],(self.n_agents,1))
        #     state_tile_mask = np.copy(state_tile)
        #     np.fill_diagonal(state_tile_mask, 0)
        #     state_obs[:,axis*self.n_agents:(axis+1)*self.n_agents] = np.copy(state_tile.T - state_tile_mask) / self.map_lim

        # final_obs = np.c_[self.battery_status, state_obs]

        self.prize_map = np.zeros(self.map_grids.shape[0])

        for i in range(self.N_prize):
            self.prize_locations[i] = self.map_grids[self.prize_grids[i]]
            if self.prize_exists[i]:  
                closest_grids = self.get_closest_n_grids(self.prize_locations[i], 4)
                self.prize_map[closest_grids] = 1

        self.prize_map = self.prize_map.reshape(1, self.out_shape, self.out_shape)
        
        return self.prize_map



    def reset(self, seed_number):
        self.agents = []
        self.prize_map = np.zeros(self.map_grids.shape[0])
        self.grid_points = []
        
        #There will be two obstacles around (x1,x2,y1,y2)=(-9,-7,5,16) and (x1,x2,y1,y2)=(7,9,-10,10) with -+ 3m deviation in x and y 
        x_rnd = 0 #np.random.uniform(-3,3)
        y_rnd = 0 #np.random.uniform(-3,3)
        # self.obstacle_start = np.array([[-9+x_rnd,5+y_rnd,0],[7+x_rnd, -10+y_rnd,0]]) 
        # self.obstacle_end = np.array([[-7+x_rnd,16+y_rnd,6],[9+x_rnd,10+y_rnd,6]])

        # Points of grids to be drawn
        x_list = np.arange(-self.map_lim, self.map_lim, self.grid_res/2.0)
        y_list = np.arange(-self.map_lim, self.map_lim, self.grid_res/2.0)
        eps = 0.01
        for x in x_list:
            grid = [x, -self.map_lim, x+eps, self.map_lim]
            self.grid_points.append(grid)
            
        for y in y_list:
            grid = [-self.map_lim, y, self.map_lim, y+eps]
            self.grid_points.append(grid) 

        # Initialization points of agents
        self.init_pos_list = []
        init_grid_list = []
        agent_pos_lim = self.map_lim - 2.0
        for i in range(5):
            init_grid_list.append(self.get_closest_grid([-agent_pos_lim + 2*i, -agent_pos_lim]))
            self.init_pos_list.append([-agent_pos_lim + 2*i, -agent_pos_lim])
        for i in range(1,6):
            init_grid_list.append(self.get_closest_grid([-agent_pos_lim , -agent_pos_lim + 2*i]))
            self.init_pos_list.append([-agent_pos_lim , -agent_pos_lim + 2*i])

        

        self.obstacle_indices = self.get_obstacle_indices()
        total_indices = np.arange(self.map_grids.shape[0])
        self.no_obstacle_indices = np.setdiff1d(total_indices, self.obstacle_indices)
        self.open_indices = np.setdiff1d(self.no_obstacle_indices, init_grid_list)

        
        #Initialization points of prizes
        np.random.seed(seed_number)
        self.N_prize = np.random.randint(1,10)
        self.prize_grids = np.random.choice(self.open_indices, (self.N_prize,), replace=False)
        self.prize_exists = np.ones(self.N_prize, dtype=bool)
        self.prize_locations = np.zeros((self.N_prize,2))
        

        return self.get_observation()
        
    
    def create_trajectory(self, agent_ind):
        n_left_prizes = np.sum(self.prize_exists)
        euclidean_dist = 1 / (np.sum((self.agents[agent_ind].state - self.prize_locations)**2,axis=1) + 1)
        euclidean_dist[~self.prize_exists] = 0
        # print ("\n \n n_left_prizes: ", n_left_prizes)
        # print ("euclidean_dist: ", 1 / euclidean_dist)
        mean_euclidean_dist = np.mean(euclidean_dist)
        # euclidean_dist[euclidean_dist < mean_euclidean_dist] *= self.n_agents # encouraging effect
        sum_euclidean_dist = np.sum(euclidean_dist)
        prob_values = euclidean_dist / sum_euclidean_dist   
        # print ("prob_values: ", prob_values)     
        target_prize = np.random.choice(np.arange(self.N_prize),p=prob_values)
        # print ("traj - self.N_prize: ", self.N_prize)
        # print ("traj - target_prize: ", target_prize)
        # print ("traj - assigned_agents_to_prizes: ", self.assigned_agents_to_prizes)
        self.assigned_agents_to_prizes[target_prize].append(agent_ind)
    
    
        path, action_list = self.astar_agent(self.agents[agent_ind].state, self.prize_locations[target_prize])
        return action_list[1:]

    def check_collision(self, agent_grid):
        s = set(self.obstacle_indices)
        if agent_grid in s:
            # print ("collided grid: ", index)
            # print ("collided grid position: ", self.map_grids[index])
            return True

        return False

    def get_agent_desired_grid(self, agent_index, discrete_action):
        agent_prev_state = np.copy(self.agents[agent_index].state)

        if discrete_action == 0: #action=0, x += 1.0
            self.agents[agent_index].state[0] += 2*self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], -self.map_lim,  self.map_lim)
        elif discrete_action == 1: #action=1, x -= 1.0
            self.agents[agent_index].state[0] -= 2*self.grid_res
            self.agents[agent_index].state[0] = np.clip(self.agents[agent_index].state[0], -self.map_lim,  self.map_lim)
        elif discrete_action == 2: #action=2, y += 1.0
            self.agents[agent_index].state[1] += 2*self.grid_res
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], -self.map_lim,  self.map_lim)
        elif discrete_action == 3: #action=3, y -= 1.0
            self.agents[agent_index].state[1] -= 2*self.grid_res
            self.agents[agent_index].state[1] = np.clip(self.agents[agent_index].state[1], -self.map_lim,  self.map_lim)
        elif discrete_action == -1: #action=-1 stop
            print ("No action executed!")
        else:
            print ("Invalid discrete action!")

        agent_current_state = np.copy(self.agents[agent_index].state)
        return agent_prev_state, agent_current_state


    def get_closest_n_grids(self, current_pos, n):
        differences = current_pos-self.map_grids
        distances = np.sum(differences*differences,axis=1)
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        
        return sorted_indices[0:n]

    def get_closest_grid(self, current_pos):
        differences = current_pos-self.map_grids
        distances = np.sum(differences*differences,axis=1)
        min_ind = np.argmin(distances)
        
        return min_ind


    def get_obstacle_indices(self):
        grid_res = 1.0
        obstacle_indices = []
        obstacle_indices_unsquezed = []
        
        # obstacle_start = np.array([[-20, -20], [-20, -20], [19, -20], [-20, 19]]) 
        # obstacle_end = np.array([[-19, 20], [20, -19], [20, 20], [20, 20]])

        for i in range(self.obstacle_start.shape[0]):
            x_range = np.arange(self.obstacle_start[i,0], self.obstacle_end[i,0], grid_res)
            y_range = np.arange(self.obstacle_start[i,1], self.obstacle_end[i,1], grid_res)
            indices = []
            for x in x_range:
                for y in y_range:
                    current_pos = np.array([x,y])
                    current_ind = self.get_closest_grid(current_pos)
                    if current_ind not in indices:
                        indices.append(current_ind)

            obstacle_indices.append(indices)
            
        for i in range(len(obstacle_indices)):
            for j in range(len(obstacle_indices[0])):
                obstacle_indices_unsquezed.append(obstacle_indices[i][j])

        return obstacle_indices_unsquezed


    def visualize(self, agent_pos_dict=None, mode='human'):
        grids = []
        station_transform = []

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.map_lim,
                                   self.map_lim, -self.map_lim, self.map_lim)
            fname = path.join(path.dirname(__file__), "assets/drone.png")
            fname_prize = path.join(path.dirname(__file__), "assets/prize.png")

            background = rendering.make_polygon([(-self.map_lim, -self.map_lim),(-self.map_lim, self.map_lim), 
                                                 (self.map_lim, self.map_lim), (self.map_lim, -self.map_lim)])

            background_transform = rendering.Transform()
            background.add_attr(background_transform)
            background.set_color(0., 0.9, 0.5) #background color
            self.viewer.add_geom(background)


            # obstacle_pos_xy = [x_min, y_min, z_min, x_max, y_max, z_max]
            obstacle_vis_start = np.array([[-20, -20], [-20, -20], [19, -20], [-20, 19]]) 
            obstacle_vis_end = np.array([[-19, 20], [20, -19], [20, 20], [20, 20]])
            for i in range(obstacle_vis_start.shape[0]):
                obstacle = rendering.make_polygon([(obstacle_vis_start[i][0], obstacle_vis_start[i][1]), 
                                                    (obstacle_vis_start[i][0], obstacle_vis_end[i][1]), 
                                                    (obstacle_vis_end[i][0], obstacle_vis_end[i][1]), 
                                                    (obstacle_vis_end[i][0], obstacle_vis_start[i][1])])

                obstacle_transform = rendering.Transform()
                obstacle.add_attr(obstacle_transform)
                obstacle.set_color(.8, .3, .3) #obstacle color
                self.viewer.add_geom(obstacle)

            for j in range(len(self.grid_points)):
                grids.append(rendering.make_polygon([(self.grid_points[j][0],self.grid_points[j][1]), 
                                                    (self.grid_points[j][0],self.grid_points[j][3]), 
                                                    (self.grid_points[j][2],self.grid_points[j][3]), 
                                                    (self.grid_points[j][2],self.grid_points[j][1])]))

                station_transform.append(rendering.Transform())
                grids[j].add_attr(station_transform[j])
                grids[j].set_color(.1, .5, .8) #grid color
                self.viewer.add_geom(grids[j])

            self.agent_transforms = []
            self.agents_img = []
            self.prizes = []
            self.prize_transformations = []

            for i in range(10):
                self.agent_transforms.append(rendering.Transform())
                self.agents_img.append(rendering.Image(fname, 1.5, 1.5)) # agent size
                self.agents_img[i].add_attr(self.agent_transforms[i])

            for i in range(10):
                self.prize_transformations.append(rendering.Transform())
                self.prizes.append(rendering.Image(fname_prize, 2., 2.)) # prize size
                self.prizes[i].add_attr(self.prize_transformations[i])

        
        
        for i in range(self.n_agents):
            self.viewer.add_onetime(self.agents_img[i])
            self.agent_transforms[i].set_translation(self.agents[i].state[0], self.agents[i].state[1])
            self.agent_transforms[i].set_rotation(0)

        for i in range(self.N_prize):
            if self.prize_exists[i] == True:
                self.viewer.add_onetime(self.prizes[i])
                self.prize_transformations[i].set_translation(self.prize_locations[i][0], self.prize_locations[i][1])
                self.prize_transformations[i].set_rotation(0)
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def astar_agent(self, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""
        x_lim, y_lim = 20, 20
        res = 2.0

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0
        visited_grids = []
        action_list = []

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append([current.position, current.action])
                    action_list.append(current.action)
                    current = current.parent
                return path[::-1], action_list[::-1] # Return reversed path and visited grids

            # Generate children
            children = []
            for index, new_position in enumerate([(res, 0), (-res, 0), (0, res), (0, -res)]): # Adjacent squares
    #         for new_position in [(0, -grid_res, 0), (0, grid_res, 0), (-grid_res, 0, 0), (grid_res, 0, 0)]: # Adjacent squares
                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
    #             print ("Node position: ", node_position)
                node_index = self.get_closest_grid(node_position)
    #             print ("Node index: {0} Node pos: {1}/{2}".format(node_index, map_grids[node_index], node_position))
                
                
                # Make sure within range
                if node_position[0] > x_lim or node_position[0] < -x_lim or node_position[1] > y_lim or node_position[1] < -y_lim:
    #                 print ("It's not within the range. Node position: ", node_position)
                    continue
                
                    
                if node_index in self.obstacle_indices:
    #                 print ("It's a obstacle place. Node position: ", node_position)
                    continue
                        
                    

                # Create new node
                new_node = Node(current_node, node_position, index)
                
                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                
                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue
                # Add the child to the open list
                open_list.append(child)



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
