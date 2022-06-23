import os
import operator
import gym
import numpy as np
import torch
import torch.nn as nn
from gym.utils import seeding
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from point_mass_env import AgentFormation

obs_space = ["1d OBS / MLP", "2D OBS / MLP", "1d OBS / CNN", "2D OBS / CNN"]
activation_list = [nn.ReLU, nn.LeakyReLU, nn.Tanh]
gamma_list = [0.9, 0.95, 0.97]
bs_list = [16, 32, 64]
lr_list = [6e-3, 9e-3, 3e-4]
net_list = [[32, 32], [48,48], [64, 64]]
ns_list = [2048, 2048, 2048]
ne_list = [12, 15, 10]

def create_map_list(prob_map, map_lim, N_maps, seed):
    new_maps = []
    np_random, _ = seeding.np_random(seed)
    for i in range(N_maps):
        map_0 = np.zeros((map_lim, map_lim))
        for r in range(map_lim):
            for c in range(map_lim):
                map_0[r][c] = np_random.choice(np.arange(0, 3), p=prob_map[r,c])
        new_maps.append(map_0)
    return new_maps


def create_prob_list(map_indices, map_lists,map_lim):
    prob_map = dict()
    for r in range(map_lim):
        for c in range(map_lim):
            key_dict = {0:0, 1:0, 2:0}
            for i in map_indices:
                key_dict[map_lists[i][r][c]] += 1
            prob_map[(r,c)] = np.array(list(key_dict.values())) / np.sum(list(key_dict.values()))
    return prob_map

def get_map(prize, obstacle, map_lim):
    generated_map = np.zeros((map_lim, map_lim)) 
    for o in obstacle:
        o_r = o // map_lim
        o_c = o % map_lim
        generated_map[o_r][o_c] = 1
    for p in prize:
        p_r = p // map_lim
        p_c = p % map_lim
        generated_map[p_r][p_c] = 2
    
    return generated_map

def generate_maps(N_maps=1, map_lim=10):
    gen_map_list = []
    
    p1 = 0.72  #np.random.uniform(0.65, 0.8)
    p2 = 0.03 #np.random.uniform(0.025, 0.1)
    for i in range(N_maps):
        gen_map = np.random.choice(3, (map_lim,map_lim), p=[p1, 1-p1-p2, p2])
        gen_map_list.append(gen_map)

    return gen_map_list


def curriculum_design(gen_map_list, rng, coeff=1.0):
    modified_map_list = []

    for gen_map in gen_map_list:
        obstacles = np.argwhere(gen_map == 1)
        rewards = np.argwhere(gen_map == 2)
        modified_map = np.copy(gen_map)

        n_samples = len(obstacles) - int(len(obstacles) * coeff)
        obstacle_to_remove = rng.choice(obstacles, size=(n_samples,), replace=False)
        for obs_loc in obstacle_to_remove:
            modified_map[obs_loc[0], obs_loc[1]] = 0
        modified_map_list.append(modified_map)
        
    return modified_map_list


def init_network(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
        
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
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(), nn.Flatten())
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.size()) == 3:
            observations = torch.reshape(observations, (1, *observations.size()))
        return self.linear(self.cnn(observations))



class CNN_Network(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNN_Network, self).__init__(observation_space, features_dim)
        init_ = lambda m: init_network(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), nn.init.calculate_gain('relu')) 
        n_input_channels = observation_space.shape[0]

        self.feat_extract = nn.Sequential(
                init_(nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(),
                init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                    nn.ELU(), nn.Flatten()
                )
        
        with torch.no_grad():
            n_flatten = self.feat_extract(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            init_(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
            init_(nn.Linear(features_dim, features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.size()) == 3:
            observations = torch.reshape(observations, (1, *observations.size()))

        cnn_out = self.feat_extract(observations)
        lin_out = self.linear(cnn_out)
        return lin_out


def print_coeffs(folder_name, obs_index, index):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(folder_name + '/coeffs.txt','w') as out:
        out.write(f" obs_space: {obs_space[obs_index]}\n activation: {str(activation_list[index])}\n gamma: {gamma_list[index]:.5}\n batch_size: {bs_list[index]}\n learning_rate: {lr_list[index]:.5}\n network: {net_list[index]}\n n_steps: {ns_list[index]}\n n_epochs: {ne_list[index]}\n")


def make_env(gen_map: np.array, max_steps = 5000):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init():
        env = AgentFormation(generated_map=gen_map, max_steps = max_steps)
        return env
    # set_random_seed(seed)
    return _init

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, subfolder_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, subfolder_dir)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-1000:])
              self.logger.record('mean_reward', mean_reward)
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}, Length of y: {len(y)}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True