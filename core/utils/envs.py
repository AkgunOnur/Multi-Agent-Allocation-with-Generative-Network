from typing import Callable, Iterable
import os
os.environ['GYM_PASSIVE_ENV_CHECKER'] = 'disable'

import gym
from gym.spaces import Box
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.transform_observation import TransformObservation
from gym_minigrid.envs.multiroom import MultiRoomEnv
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class MinigridWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int = 4,
        seed: int = -1,
    ):
        super(MinigridWrapper, self).__init__(env)

        self._seed = seed if seed != -1 else None
        self.num_stack = num_stack

        # only keep image as state
        env = ImgObsWrapper(env)
        # env = RGBImgPartialObsWrapper(env)
        print ("observation_space: ", env.observation_space)

        # define new state space shape
        ob_shape = env.observation_space.shape
        stacked_obs_shape = (ob_shape[-1] * num_stack,) + ob_shape[:-1]

        # stack frames
        # env = FlattenObservation(env)
        env = FrameStack(env=env, num_stack=num_stack)

        # transpose and reshape observation
        reshape_obs = (
            lambda obs: obs[:].transpose(0, -1, 1, 2).reshape(stacked_obs_shape)
        )
        env = TransformObservation(env, f=reshape_obs)

        # set new env, action and observation spaces
        ob_space = env.observation_space
        env.observation_space = Box(
            low=reshape_obs(ob_space.low),
            high=reshape_obs(ob_space.high),
            shape=stacked_obs_shape,
            dtype=ob_space.dtype,
        )

        # set env
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        if self._seed:
            self.env.seed(self._seed)
        observation = self.env.reset()
        # start with empty observations
        observation[: (self.num_stack - 1) * 3] = 0
        return observation


class EpisodicCountWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, state_key_extraction: Callable[..., Iterable]):
        super(EpisodicCountWrapper, self).__init__(env)
        self.state_key_extraction = state_key_extraction


class MultiRoomEnvN10S10(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=10, maxNumRooms=10, maxRoomSize=10)


class MultiRoomEnvN10S6(MultiRoomEnv):
    def __init__(self):
        super().__init__(minNumRooms=10, maxNumRooms=10, maxRoomSize=6)
