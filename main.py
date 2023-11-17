import argparse
import pickle
from baselines import logger

from rapid.train import train
import warnings

# Ignore specific FutureWarnings related to NumPy dtype deprecations
warnings.filterwarnings("ignore", message="Passing (type, 1) or '1type' as a synonym of type is deprecated")

def argparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of RAPID")
    # parser.add_argument('--env', help='environment ID', type=str, default='MiniGrid-MultiRoom-N7-S4-v0'),
    parser.add_argument('--env', help='environment ID', type=str, default='MiniGrid-CustomDoorKey-8x8-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_timesteps', help='Number of timesteps', type=int, default=int(2e7))
    parser.add_argument('--N_train_iter', help='Number of timesteps', type=int, default=1)
    parser.add_argument('--map_file', default="doorkey_all_maps.pickle", type=str, help='name of the policy')
    parser.add_argument('--nsteps', help='nsteps', type=int, default=128)
    parser.add_argument('--log_dir', help='the directory to save log file', type=str, default='log')
    parser.add_argument('--lr', help='the learning rate', type=float, default=1e-4)
    parser.add_argument('--w0', help='weight for extrinsic rewards', type=float, default=1.0)
    parser.add_argument('--w1', help='weight for local bonus', type=float, default=0.1)
    parser.add_argument('--w2', help='weight for global bonus', type=float, default=0.001)
    parser.add_argument('--buffer_size', help='the size of the ranking buffer', type=int, default=10000)
    parser.add_argument('--batch_size', help='the batch size', type=int, default=256)
    parser.add_argument('--sl_until', help='SL until which timestep', type=int, default=100000000)
    parser.add_argument('--disable_rapid', help='Disable SL, i.e., PPO', action='store_true')
    parser.add_argument('--sl_num', help='Number of updated steps of SL', type=int, default=5)
    return parser

def main():
    parser = argparser()
    args = parser.parse_args()
    map_number = 2

    with open(args.map_file, 'rb') as handle:
        new_target_maps = pickle.load(handle)

    for j_map, target_map in enumerate(new_target_maps[map_number:]):
        for i_iter in range(args.N_train_iter):
            print ("Current Iter #" + str(i_iter) +" Env. Map #" + str(j_map) + "\n")
            print (target_map)

            args.log_dir = "log/DoorKey/map_" + str(j_map) + "_iter_" + str(i_iter) 

            logger.configure(dir=args.log_dir)
            train(args, target_map)

if __name__ == '__main__':
    main()
