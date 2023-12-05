import os
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union, Sequence
import multiprocessing as mp

# import envpool
import gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper, VecMonitor
# from envpool.python.protocol import EnvPool
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, tile_images
from gym.core import Wrapper

# From Stable Baseline 3
# https://github.com/DLR-RM/stable-baselines3/blob/18f4e3ace084a2fd3e0a3126613718945cf3e5b5/stable_baselines3/common/env_util.py

from packaging import version
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class ObservationUpdatedWrapper(Wrapper):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to the observation that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    defined on the base environment’s observation space. However, it may take values in a different space.
    In that case, you need to specify the new observation space of the wrapper by setting :attr:`self.observation_space`
    in the :meth:`__init__` method of your wrapper.

    For example, you might have a 2D navigation task where the environment returns dictionaries as observations with
    keys ``"agent_position"`` and ``"target_position"``. A common thing to do might be to throw away some degrees of
    freedom and only consider the position of the target relative to the agent, i.e.
    ``observation["target_position"] - observation["agent_position"]``. For this, you could implement an
    observation wrapper like this::

        class RelativePosition(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

            def observation(self, obs):
                return obs["target"] - obs["agent"]

    Among others, Gym provides the observation wrapper :class:`TimeAwareObservation`, which adds information about the
    index of the timestep to the observation.
    """

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observation, reward, terminated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, info

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError
    
class ImgObsWrapper(ObservationUpdatedWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        # info = {}
        return obs['image']

# class EnvPoolVecAdapter(VecEnvWrapper):
#     """
#     Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
#     :param venv: The envpool object.
#     """

#     def __init__(self, venv: EnvPool):
#         # Retrieve the number of environments from the config
#         venv.num_envs = venv.spec.config.num_envs
#         super().__init__(venv=venv)
#         self.venv.obs = None

#     def step_async(self, actions: np.ndarray) -> None:
#         self.actions = actions

#     def reset(self) -> VecEnvObs:
#         if is_legacy_gym:
#             obs = self.venv.reset()
#         else:
#             obs = self.venv.reset()[0]
#         self.venv.obs = obs
#         return obs

#     def seed(self, seed: Optional[int] = None) -> None:
#         # You can only seed EnvPool env by calling envpool.make()
#         pass

#     def step_wait(self) -> VecEnvStepReturn:
#         if is_legacy_gym:
#             obs, rewards, dones, info_dict = self.venv.step(self.actions)
#         else:
#             obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
#             dones = terms + truncs

#         infos = []
#         # Convert dict to list of dict
#         # and add terminal observation
#         for i in range(self.num_envs):
#             infos.append(
#                 {
#                     key: info_dict[key][i]
#                     for key in info_dict.keys()
#                     if isinstance(info_dict[key], np.ndarray)
#                 }
#             )
#             if dones[i]:
#                 infos[i]["terminal_observation"] = obs[i]
#                 if is_legacy_gym:
#                     obs[i] = self.venv.reset(np.array([i]))
#                 else:
#                     obs[i] = self.venv.reset(np.array([i]))[0]
#         self.venv.obs = obs
#         return obs, rewards, dones, infos

#     def render(self, mode: str = "human") -> Optional[np.ndarray]:
#         if self.venv.obs is None:
#             return

#         try:
#             imgs = self.venv.obs
#         except NotImplementedError:
#             warnings.warn(f"Render not defined for {self}")
#             return

#         # Create a big image by tiling images from subprocesses
#         bigimg = tile_images(imgs[:1])

#         bigimg_size = bigimg.shape[-1]
#         bigimg = bigimg[-1].reshape(bigimg_size, bigimg_size)

#         # grayscale to fake-RGB
#         bigimg = np.stack((bigimg,) * 3, axis=-1)

#         if mode == "human":
#             import cv2  # pytype:disable=import-error
#             cv2.imshow("vecenv", bigimg[:, :, ::-1])
#             cv2.waitKey(1)
#         elif mode == "rgb_array":
#             return bigimg
#         else:
#             raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

