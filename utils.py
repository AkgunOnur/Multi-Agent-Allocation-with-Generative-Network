import os
import numpy as np
import torch.nn as nn
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