import gym
from gym import spaces
from gym.utils import seeding
#from gym.envs.classic_control import rendering
import numpy as np
from os import path
from environment.quadrotor_dynamics import Drone
from numpy.random import uniform
from time import sleep


class QuadrotorFormation(gym.Env):

    def __init__(self, n_agents=1, visualization=False):

        self.seed()
        self.n_action = 8
        self.visualization = visualization

        self.agent = None
        self.viewer = None
        
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(4, 20, 20), dtype=np.uint8)


        self.x_lim = 20  # grid x limit
        self.y_lim = 20  # grid y limit

        self.wall_map = np.zeros((self.x_lim, self.y_lim))
        self.reward_map = np.zeros((self.x_lim, self.y_lim))
        self.path_map = np.zeros((self.x_lim, self.y_lim))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Environment Step
        done = False
        reward = -0.1

        self.update_agent_pos(action)

        if int(self.reward_map[self.agent.x, self.agent.y]) == 1:
            self.reward_map[self.agent.x, self.agent.y] = 0
            reward += 0.8
        
        if np.all(self.reward_map == 0):
            done = True

        if self.visualization:
            self.render()

        state = self.get_observation()

        return state, reward, done, _

    def get_observation(self):

        state = np.zeros((4,20,20))

        state[0] = self.path_map
        state[1] = self.wall_map
        state[2] = self.reward_map
        state[3] = self.agent.state

        return state


    def generate_agent_position(self, agentX, agentY):
        state0 = np.zeros = ((20,20))

        state0[agentX, agentY] = 1.0

        self.agent = Drone(state0)
        self.agent.x = agentX
        self.agent.y = agentY


    def reset(self, observation_map):

        agent_initX = 2
        agent_initY = 2
        self.path_map = observation_map[0]
        self.wall_map = observation_map[1]
        self.reward_map = observation_map[2]
        
        self.generate_agent_position(agent_initX, agent_initY)
        self.iteration = 1

        state = self.get_observation()

        return state


    def update_agent_pos(self, discrete_action):
        
        prev_state = self.agent.state

        self.agent.state = np.zeros((20,20))

        if discrete_action == 0: # action=0, x += 1.0
            self.agent.x += 1.0
            self.agent.x = np.clip(self.agent.x, 0,  self.x_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 1: # action=1, x -= 1.0
            self.agent.x -= 1.0
            self.agent.x = np.clip(self.agent.x, 0,  self.x_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 2: # action=2, y += 1.0
            self.agent.y += 1.0
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 3: # action=3, y -= 1.0
            self.agent.y -= 1.0
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 4: # action=4, x,y += 1.0
            self.agent.x += 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agentx, 0,  self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 5: # action=5, x,y -= 1.0
            self.agent.x -= 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0,  self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 6: # action=6, x += 1.0 ,y -= 1.0
            self.agent.x += 1.0
            self.agent.y -= 1.0
            self.agent.x = np.clip(self.agent.x, 0,  self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        elif discrete_action == 7: # action=7, x -= 1.0 ,y += 1.0
            self.agent.x -= 1.0
            self.agent.y += 1.0
            self.agent.x = np.clip(self.agent.x, 0,  self.x_lim)
            self.agent.y = np.clip(self.agent.y, 0,  self.y_lim)
            self.agent.state[self.agent.x, self.agent.y] = 1.0

        else:
            print ("Invalid discrete action!")

        if int(self.wall_map[self.agent.x, self.agent.y]) == 1:
            self.agent.state = prev_state


    def render(self, mode='human'):
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
