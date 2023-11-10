import argparse
import pickle
from stable_baselines3 import A2C, PPO, DQN, SAC, DDPG, TD3
# from sb3_contrib import ARS, MaskablePPO, TQC, RecurrentPPO, QRDQN
# from curriculum_bayes import bayes_curriculum_april, vanilla_train
# from curriculum_june import bayes_curriculum_june
# from curriculum_test import *
# from curriculum_august import *
from curriculum_october import bayes_curriculum_october
# from autoencoder_train import *
# from curriculum_procgen import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--env_id', default="", type=str, help='scenario') #MiniGrid-CustomDoorKey-8x8-v0, MiniGrid-CustomKeyCorridor-v0, MiniGrid-ObstructedMaze-2Dlhb-v0, MiniGrid-MultiRoom-N6-v0, MiniGrid-KeyCorridorS4R3-v0, MiniGrid-KeyCorridorS5R3-v0
    parser.add_argument('--output_folder', default="7Kasim_DoorKey_Curriculum", type=str, help='output folder name')
    parser.add_argument('--target_index', default=2, type=int, help='target index')
    parser.add_argument('--env_type', default="DoorKey", type=str, help='name of the environment')
    parser.add_argument('--map_file', default="doorkey_all_maps.pickle", type=str, help='name of the policy')
    parser.add_argument('--autoencoder_model', default="autoencoder_August3_doorkey/best_model.pth", type=str, help='encoder model to be utilized')
    parser.add_argument('--N_train_iter', default=5, type=int, help='train iteration')
    # parser.add_argument('--tsne_train_file', default="extracted_features_keycorridor_1_resnet152_normal.pickle", type=str, help='folder name')
    parser.add_argument('--train_timesteps', default=100000000, type=int, help='train timesteps')
    parser.add_argument('--seed', default=1, type=int, help='seed number for test')
    parser.add_argument('--n_eval_freq', default=400, type=int, help='evaluation interval')
    parser.add_argument('--n_eval_episodes', default=10, type=int, help='evaluation episodes')
    parser.add_argument('--n_procs', default=4, type=int, help='number of processes to execute')
    parser.add_argument('--deterministic', default = True, type=bool, help='how should the policy act')
    parser.add_argument('--algorithm', default=PPO, help='name of the algorithm')
    parser.add_argument('--policy', default="CnnPolicy", type=str, help='name of the policy')
    # parser.add_argument('--episode_length', default = 2500, type=int, help='episode length')
    parser.add_argument('--load_folder', default="", type=str, help='output folder name')
    parser.add_argument('--visualization', default = False, type=bool, help='to visualize')
    args = parser.parse_args()


    if args.env_type == "KeyCorridor":
        args.env_id = "MiniGrid-CustomKeyCorridor-v0"
    elif args.env_type == "DoorKey":
        args.env_id = "MiniGrid-CustomDoorKey-v0"

    bayes_curriculum_october(args)
    # vanilla_train(args=args)

    # create_dataset(args)
    # train_autoencoder(args)
    
    # for i_iter in range(args.N_train_iter):
    #     args.output_folder = folder_name + "_iter_" + str(i_iter)
    #     bayes_curriculum_august(args)
    # bayes_curriculum_june(args=args)
    # create_dataset(args)
    # train_autoencoder(args)
    # bayes_curriculum_april(args=args)
    # bayes_curriculum_test(args=args)
    # randomized_train(args=args)
    # manual_curriculum(args)
    # new_bayes_train(args)
    
    # noncurriculum_train(args=args)
    