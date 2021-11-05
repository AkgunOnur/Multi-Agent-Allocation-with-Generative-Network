from numpy.core.arrayprint import dtype_short_repr
import gym
import pickle
import os
import time
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
from environment.quadrotor_dynamics import Drone
from numpy.random import uniform
from time import sleep
from PIL import Image

class QuadrotorFormation(gym.Env):

    def __init__(self, map_type, visualization=False):
        super(QuadrotorFormation, self).__init__()

        self.seed()
        self.n_action = 8
        self.visualization = visualization

        self.agent = None
        self.viewer = None
        self.rewPos_list = []
        self.wallPos_list = []
        self.map_type = map_type
        
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(20, 20, 4), dtype=np.uint8)

        self.x_lim = 19
        self.y_lim = 19
        self.iteration = 0

        self.wall_map = np.zeros((self.y_lim, self.x_lim))
        self.reward_map = np.zeros((self.y_lim, self.x_lim))
        self.path_map = np.zeros((self.y_lim, self.x_lim))

        self.map_index = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = False
        reward = -0.01
        self.iteration += 1

        self.update_agent_pos(action)

        if int(self.reward_map[int(self.agent.y), int(self.agent.x)]) == 1:
            self.reward_map[int(self.agent.y),int( self.agent.x)] = 0
            reward += 0.8

        self.reward_wall_num()
        state = self.get_observation()

        if self.visualization:
            self.render()

        if np.all(self.reward_map == 0) or self.iteration >= 600:
            done = True
            self.close()
        info = {"is_success": done}
        return state, reward, done, info

    def get_observation(self):

        state = np.zeros((20,20,4))

        state[:,:,0] = self.path_map*255.0
        state[:,:,1] = self.wall_map*255.0
        state[:,:,2] = self.reward_map*255.0
        state[:,:,3] = self.agent.state*255.0

        return np.array(state, dtype=np.uint8)


    def generate_agent_position(self, agentY, agentX):
        state0 = np.zeros((20,20))

        state0[agentY, agentX] = 1.0

        self.agent = Drone(state0)
        self.agent.x = agentX
        self.agent.y = agentY

    def get_init_map(self, index):

        if self.map_type == "gan":
            with open('training_map_library.pkl', 'rb') as f:
                map_dataset = pickle.load(f)

        elif self.map_type == "random":
            with open('training_maps_random_without_gan.pkl', 'rb') as f:
                map_dataset = pickle.load(f)

        map_dataset = np.array(map_dataset[0]).squeeze(1)   

        return map_dataset[index]

    def reward_wall_num(self):
        self.wallPos_list = []
        self.rewPos_list = []

        wall_row, wall_col = np.where(self.wall_map == 1)
        reward_row, reward_col = np.where(self.reward_map == 1)

        for x in zip(wall_row, wall_col):
            self.wallPos_list.append(x)
        for x in zip(reward_row, reward_col):
            self.rewPos_list.append(x)

    def reset(self):

        self.iteration = 0
        init_map = self.get_init_map(self.map_index)
        self.map_index += 1
        self.map_index %= 600

        agent_initX = 2
        agent_initY = 2
        self.path_map = init_map[0]
        self.wall_map = init_map[1]
        self.reward_map = init_map[2]

        self.reward_wall_num()

        self.generate_agent_position(agent_initY, agent_initX)

        state = self.get_observation()

        return state

    def update_agent_pos(self, discrete_action):
        
        prev_state = self.agent.state.copy()
        prev_x = self.agent.x
        prev_y = self.agent.y

        self.agent.state = np.zeros((20,20))

        if discrete_action == 0: # action=0, x += 1.0
            self.agent.x += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)

        elif discrete_action == 1: # action=1, x -= 1.0
            self.agent.x -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)

        elif discrete_action == 2: # action=2, y += 1.0
            self.agent.y += 1.0
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 3: # action=3, y -= 1.0
            self.agent.y -= 1.0
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 4: # action=4, x,y += 1.0
            self.agent.x += 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 5: # action=5, x,y -= 1.0
            self.agent.x -= 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 6: # action=6, x += 1.0 ,y -= 1.0
            self.agent.x += 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        elif discrete_action == 7: # action=7, x -= 1.0 ,y += 1.0
            self.agent.x -= 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agent.x, 0, self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0, self.y_lim)

        else:
            print ("Invalid discrete action!")

        if int(self.wall_map[int(self.agent.y),int(self.agent.x)]) == 1:
            self.agent.state = prev_state
            self.agent.x = prev_x
            self.agent.y = prev_y
        else:
            self.agent.state[int(self.agent.y),int(self.agent.x)] = 1.0

        # time.sleep(1)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.x_lim, 0, self.y_lim)
            # fname = path.join(path.dirname(__file__), "sprites/drone.png")
            mapsheet = Image.open(os.path.join(path.dirname(__file__), 'sprites/mapsheet.png'))

            drone_path = os.path.join(path.dirname(__file__), 'sprites/drone.png')
            reward_path = os.path.join(path.dirname(__file__), 'sprites/reward.png')
            wall_path = os.path.join(path.dirname(__file__), 'sprites/wall.png')
            road_path = os.path.join(path.dirname(__file__), 'sprites/path.png')

            sprite_dict = dict()
            sprite_dict['D'] = mapsheet.crop((4*16, 0, 5*16, 1*16))
            sprite_dict['X'] = mapsheet.crop((7*16, 1*16, 8*16, 2*16))
            sprite_dict['O'] = mapsheet.crop((2*16, 0, 3*16, 1*16))
            sprite_dict['-'] = mapsheet.crop((2*16, 5*16, 3*16, 6*16))

            sprite_dict['D'].save(drone_path)
            sprite_dict['X'].save(reward_path)
            sprite_dict['O'].save(wall_path)
            sprite_dict['-'].save(road_path)

            self.drone_transforms = []
            self.drones = []

            self.reward_transform = []
            self.render_rew = []

            self.wall_transform = []
            self.render_wall = []

            for i in range(1):
                self.drone_transforms.append(rendering.Transform())
                self.drones.append(rendering.Image(drone_path, 1., 1.))
                self.drones[i].add_attr(self.drone_transforms[i])

            for i in range(len(self.rewPos_list)):
                self.reward_transform.append(rendering.Transform())
                self.render_rew.append(rendering.Image(reward_path, 1., 1.))
                self.render_rew[i].add_attr(self.reward_transform[i])
            
            for i in range(len(self.wallPos_list)):
                self.wall_transform.append(rendering.Transform())
                self.render_wall.append(rendering.Image(wall_path, 1., 1.))
                self.render_wall[i].add_attr(self.wall_transform[i])
        
        for i in range(1):
            self.viewer.add_onetime(self.drones[i])
            self.drone_transforms[i].set_translation(self.agent.x, self.y_lim-self.agent.y) 
        
        for i in range(len(self.wallPos_list)):
            self.viewer.add_onetime(self.render_wall[i])
            self.wall_transform[i].set_translation(self.wallPos_list[i][1], self.y_lim-self.wallPos_list[i][0])
        
        for i in range(len(self.rewPos_list)):
            if not len(self.rewPos_list)==0:
                self.viewer.add_onetime(self.render_rew[i])
                self.reward_transform[i].set_translation(self.rewPos_list[i][1], self.y_lim-self.rewPos_list[i][0])
            
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
