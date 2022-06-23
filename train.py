import argparse
from stable_baselines3 import A2C, PPO, DQN
from curriculum_approaches import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--train_timesteps', default=250000, type=int, help='number of test iterations')
    parser.add_argument('--n_eval_freq', default=500, type=int, help='evaluation interval')
    parser.add_argument('--map_lim', default=10, type=int, help='map size')
    parser.add_argument('--n_procs', default=4, type=int, help='number of processes to execute')
    parser.add_argument('--n_map', default=20, type=int, help='number of maps in each generation')
    parser.add_argument('--n_population', default=10, type=int, help='number of population in each generation')
    parser.add_argument('--n_generation', default=20, type=int, help='number of generation in each iteration')
    parser.add_argument('--n_iteration', default=10, type=int, help='number of iteration in total')
    parser.add_argument('--seed', default=100, type=int, help='seed number for test')
    parser.add_argument('--algo', default=PPO, help='name of the algorithm')
    parser.add_argument('--load', default="", type=str, help='model to be loaded')
    parser.add_argument('--target_map', default="map_2", type=str, help='the target map to solve')
    parser.add_argument('--visualize', default = False, action='store_true', help='to visualize')
    args = parser.parse_args()
    
    # noncurriculum_train(folder_postfix = "12Haziran_PPO_Noncurriculum_5M", train_timesteps=args.train_timesteps, algorithm=args.algo, n_procs=args.n_procs, target_map=args.target_map, map_lim=args.map_lim, N_eval_freq=args.n_eval_freq, output_folder = args.output_folder, visualize = args.visualize)
    # cma_train(folder_postfix = "15Haziran_PPO_CMA_250k", output_folder="output_15Haziran_CMA", train_timesteps=250000, algorithm=args.algo, n_procs=args.n_procs, target_map=args.target_map, map_lim=args.map_lim, N_eval_freq=args.n_eval_freq, N_population=args.n_population, N_generation=args.n_generation, N_iteration=args.n_iteration, seed = args.seed, visualize = args.visualize)
    randomized_train(folder_postfix = "22Haziran_PPO_250k", output_folder="output_22Haziran_Randomized", train_timesteps=250000, algorithm=args.algo, n_procs=args.n_procs, target_map=args.target_map, map_lim=args.map_lim, N_eval_freq=args.n_eval_freq, N_maps=args.n_map, N_generation=args.n_generation, N_iteration=args.n_iteration, seed = args.seed, visualize = args.visualize)
