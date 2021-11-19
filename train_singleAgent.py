import pickle
import torch
import numpy as np
import torch.nn as nn
import gym

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from environment.singleAgentEnv import QuadrotorFormation
from environment.level_utils import read_level
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


max_steps = 7200000


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),)


# Train with GAN Generated Maps
def main():
    #vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="gan"), n_envs=1, vec_env_cls=SubprocVecEnv)
    #model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, exploration_fraction = 0.8, verbose=1, tensorboard_log="./dqn_tensorboard/")
    #model = A2C('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./dqn_tensorboard/")

    #model.learn(total_timesteps=max_steps)
    #model.save("./weights/dqn_gan")

    # Train with GAN Random Maps
    vecenv = make_vec_env(lambda: QuadrotorFormation(map_type="random"), n_envs=1, vec_env_cls=SubprocVecEnv)
    model = DQN('CnnPolicy', vecenv, policy_kwargs=policy_kwargs, exploration_fraction = 0.8, verbose=1, tensorboard_log="./dqn_tensorboard/")

    model.learn(total_timesteps=max_steps)
    model.save("./weights/dqn_random")
    
if __name__ == '__main__':
    main()

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render()