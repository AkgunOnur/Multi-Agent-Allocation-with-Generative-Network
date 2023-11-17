import os
import sys
import gym
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng
from rapid.agent import Model
from rapid.utils import MlpPolicy, RGBImgObsWrapper, make_custom_env
from gym_minigrid.envs.custom_doorkey import CustomDoorKeyEnv
from baselines import bench, logger
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv





def test():
    parser = argparse.ArgumentParser(description='RL trainer')
    parser.add_argument('--env', default="MiniGrid-CustomDoorKey-8x8-v0", type=str, help='scenario')
    parser.add_argument('--eval_episodes', default=1, type=int, help='number of test iterations')
    parser.add_argument('--seed', default=1, type=int, help='number of seed')
    parser.add_argument('--visualization', default=True, type=bool, help='vis')
    parser.add_argument('--nsteps', help='nsteps', type=int, default=128)
    args = parser.parse_args()
    
    map_reward_list = []
    load_path = "log/iter2/checkpoints/13800"

    ent_coef=0.01
    vf_coef=0.5
    max_grad_norm=0.5
    nminibatches=4
    nsteps = args.nsteps


    target_map = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 5, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 3, 0, 2, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 3, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 4]])


    def _make_env():
        env = make_custom_env(env_id=args.env, current_map=target_map, visualization=args.visualization)
        # env = make_env(args.env)
        # env.seed(args.seed)
        return env

    env = DummyVecEnv([_make_env])

    # env = gym.make(args.env_id, **env_kwargs)

    # env = RGBImgObsWrapper(env,tile_size=env_kwargs["config"]["size"]) # Get rid of the 'mission' field
    # env = bench.Monitor(env, logger.get_dir())
    # env = DummyVecEnv(env)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    policy = MlpPolicy

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
 
    model = make_model()
    model.load(load_path)
    

    # env = RGBImgObsWrapper(CustomDoorKeyEnv(config))
    # env = make_vec_env(env_id=args.env_id, n_envs=1, env_kwargs=dict(config=config,frame_file = "frames_5"), vec_env_cls=DummyVecEnv)

    for episode in range(args.eval_episodes):
        obs = env.reset()
        map_reward = 0
        iteration = 0
        iter_list = []
        while True:
            # action, _states = model.predict(obs, deterministic = True)
            action, values, states, neglogpacs = model.step(obs)
            obs, reward, done, info = env.step(action)
            # print ("reward: ", reward)
            map_reward += reward[0]
            iteration += 1
            time.sleep(0.1)
        
            if done:
                print ("Episode: {0}, Reward: {1:.3f} in iteration: {2}".format(episode, map_reward, iteration))
                map_reward_list.append(map_reward)
                iter_list.append(iteration)
                break
    
    print ("Mean reward: {0:.3f}, Std. reward: {1:.3f}, in {2} episodes".format(np.mean(map_reward_list), np.std(map_reward_list), args.eval_episodes))

    
    # with open('results_' + file + '.pickle', 'wb') as handle:
    #     pickle.dump(total_reward_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.visualization:
        env.close()

if __name__ == '__main__':
    test()

