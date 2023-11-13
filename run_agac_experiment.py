import argparse
import pickle
import gym
from gym_minigrid.minigrid import MiniGridEnv

from agac.agac_trainer import AGAC
from agac.configs import get_config_from_yaml
from core.utils.envs import EpisodicCountWrapper, MinigridWrapper

def make_custom_env(env_id, target_map, visualization=False):
    seed = 0
    map_lim = target_map.shape[0]
    config = {
            "seed":seed,
            "size":map_lim + 2, "env_map":target_map, "custom":True, "visualization":visualization}

    env_kwargs=dict(config=config)

    # env = Minigrid2Image(wrappers.FullyObsWrapper(gym.make(env_id, **env_kwargs)))
    env = gym.make(env_id, **env_kwargs)
    env = MinigridWrapper(env, num_stack=4)
    return env

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/minigrid.yaml",
        help="config",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="MiniGrid-CustomDoorKey-v0",
        help="gym env name",
    )
    parser.add_argument("--seed", type=int, default=123, help="seed number")
    args = parser.parse_args()
    return args

def state_key_extraction(env):
    return tuple(env.agent_pos)

if __name__ == "__main__":
    args = get_parser_args()

    # read and merge configs
    config = get_config_from_yaml(args.config_path)
    config.algorithm.seed = args.seed
    config.algorithm.env_name = args.env_name
    N_train_iter = 5

    map_file = "doorkey_all_maps.pickle"
    with open(map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)

    for map_index, target_map in enumerate(new_target_maps):
        for train_iter in range(N_train_iter):

            target_map = new_target_maps[-1]
            env = make_custom_env(args.env_name, target_map)

            if config.reinforcement_learning.episodic_count_coefficient > 0:
                env = EpisodicCountWrapper(env=env, state_key_extraction=state_key_extraction)

            # create trainer and train
            agac = AGAC(config, env, train_iter, map_index)
            agac.train()
