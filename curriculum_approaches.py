from fileinput import filename
import sys
import os
import torch
import operator
import gym
import time
import pickle
import argparse
import numpy as np
from numpy.random import default_rng
from gym.utils import seeding

from graph import *

from cmaes import SepCMA, CMA
from point_mass_env import AgentFormation
import torch.nn as nn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoRemarkableImprovement, StopTrainingOnNoModelImprovement

from utils import *
sys.path.insert(0,'rendering_files/')

from rendering_files.level_utils import load_level_from_text
from rendering_files.level_image_gen import LevelImageGen

def noncurriculum_train(train_timesteps:int, algorithm, n_procs:int, target_map:str, map_lim: int, N_eval_freq:int, folder_postfix: str, output_folder: str, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    agent_step_cost = 0.01
    N_eval_freq = N_eval_freq // n_procs

    map_list = []

    current_folder = output_folder + "/" +  folder_postfix + "_" + target_map
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    lvl = load_level_from_text("target_maps/" + target_map + ".txt")
    current_map = np.zeros((map_lim, map_lim))
    for r in range(map_lim):
        for c in range(map_lim):
            if lvl[r][c] == '-':
                current_map[r][c] = 0
            elif lvl[r][c] == 'W':
                current_map[r][c] = 1
            elif lvl[r][c] == 'X':
                current_map[r][c] = 2
    map_list.append(current_map)
    N_obstacle = np.sum(map_list[0] == 1)
    N_prize = np.sum(map_list[0] == 2)
    N = N_obstacle + N_prize

    max_possible_reward = N_prize - agent_step_cost * 100
    

    start = time.time()

    print (f"Noncurriculum Train \n")

    train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim, visualization=visualize), n_envs=n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim), n_envs=n_procs, vec_env_cls=SubprocVecEnv)

    train_env.reset()
    eval_env.reset()

    
    model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
    
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_possible_reward, verbose=1)

    callback = EvalCallback(eval_env=eval_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                            best_model_save_path = output_folder + "/saved_models/" + folder_postfix + "_" + target_map, deterministic=deterministic, verbose=1)
    
    model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_" + target_map, callback=callback)
    train_env.close()
    eval_env.close()
    
            
    elapsed_time = time.time() - start
    print (f"Elapsed time: {elapsed_time:.5} s.")  



def randomized_train(train_timesteps:int, algorithm, n_procs:int, map_lim: int, target_map:str, N_eval_freq:int, N_maps: int, N_generation: int, N_iteration:int, folder_postfix: str, output_folder: str, seed:int, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    N_eval_freq = N_eval_freq // n_procs

    target_map_list = []
    lvl = load_level_from_text("target_maps/" + target_map + ".txt")
    current_map = np.zeros((map_lim, map_lim))
    for r in range(map_lim):
        for c in range(map_lim):
            if lvl[r][c] == '-':
                current_map[r][c] = 0
            elif lvl[r][c] == 'W':
                current_map[r][c] = 1
            elif lvl[r][c] == 'X':
                current_map[r][c] = 2

    target_map_list.append(current_map)

    for iteration in range(N_iteration): # How many times this algorithm will work
        map_rewards = dict()
        
        prob_map = dict()
        for r in range(map_lim):
            for c in range(map_lim):
                prob_map[(r,c)] = np.array([0.5, 0.15, 0.35])

        print (f"Iteration: {iteration} - Randomized Train \n")
        best_reward_list = []
        best_map_index_list = []
        total_map_list = []

        for generation in range(N_generation):
            print ("Generation: ", generation)
            generated_map_list = create_map_list(prob_map, map_lim, N_maps, seed)
            total_map_list.append(generated_map_list)

            current_folder = output_folder + "/" +  folder_postfix + "_gen_" + str(generation)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            for index, current_map in enumerate(generated_map_list):
                # start = time.time()
                map_list = []
                map_list.append(current_map)

                train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim, visualization=visualize), n_envs=n_procs, monitor_dir=current_folder + "_map_" + str(index), vec_env_cls=SubprocVecEnv)
                eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim), n_envs=n_procs, vec_env_cls=SubprocVecEnv)
                target_env = make_vec_env(lambda: AgentFormation(generated_map=target_map_list, map_lim=map_lim, max_steps=20*map_lim), n_envs=n_procs//2, vec_env_cls=SubprocVecEnv)

                train_env.reset()
                eval_env.reset()

                if generation > 0:
                    model = algorithm.load(path=output_folder + "/saved_models/" +  folder_postfix + "_gen_" + str(generation - 1) + "_best" + "/best_model", verbose=0, only_weights = False) # + "/best_model"
                    model.tensorboard_log = "./" + current_folder + "_tensorboard/"
                    model.set_env(train_env)
                    print ("Best model in gen #", (generation - 1), " is uploaded!")
                else:
                    model = algorithm('MlpPolicy', train_env, tensorboard_log= "./" + current_folder + "_tensorboard/")
                
                stop_callback = StopTrainingOnNoRemarkableImprovement(max_no_improvement_evals = 100, check_percentage=0.9, verbose = 1)
                

                callback = EvalCallback(eval_env=eval_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                        best_model_save_path = output_folder + "/saved_models/" + folder_postfix + "_gen_" + str(generation) + "_map_" + str(index), deterministic=deterministic, verbose=1)
                
                model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_gen_" + str(generation) + "_map_" + str(index), callback=callback)
                # elapsed_time = time.time() - start
                # print (f"Elapsed time: {elapsed_time:.5} s.")        
                train_env.close()


                # Load the model
                model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_gen_" + str(generation) + "_map_" + str(index) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
                eval_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                mean_reward_current = np.mean(episode_rewards)
                std_reward_current = np.std(episode_rewards)
                print (f"Mean reward in eval env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
                
                target_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=50, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                mean_reward_target = np.mean(episode_rewards)
                std_reward_target = np.std(episode_rewards)
                print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} ")

                eval_env.close()
                target_env.close()

                map_rewards[index] = mean_reward_target



            reward_sorted = np.array(sorted(map_rewards.items(), key=operator.itemgetter(1), reverse=True))

            prob_map = create_prob_list(reward_sorted[0:10,0].astype(np.int16), generated_map_list, map_lim)


            best_index = int(reward_sorted[0][0])
            best_reward_list.append(reward_sorted[0][1])
            best_map_index_list.append(best_index)

            model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_gen_" + str(generation) + "_map_" + str(best_index) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
            model.save(path=output_folder + "/saved_models/" +  folder_postfix + "_gen_" + str(generation) + "_best/best_model")

            with open(folder_postfix + "_randomized_iter_" + str(iteration) + ".pickle", 'wb') as handle:
                pickle.dump([total_map_list, best_reward_list, best_map_index_list], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
                
        # elapsed_time = time.time() - start
        # print (f"Iteration: {iteration} Elapsed time: {elapsed_time:.5} s.")  

def cma_train(train_timesteps:int, algorithm, n_procs:int, map_lim: int, target_map:str, N_eval_freq:int, N_population: int, N_generation: int, N_iteration:int, folder_postfix: str, output_folder: str, seed:int, visualize: bool):
    deterministic = True
    N_eval_episodes = 10
    N_iteration = 20
    N_generation = 20
    agent_step_cost = 0.01
    N_eval_freq = N_eval_freq // n_procs

    target_map_list = []
    lvl = load_level_from_text("target_maps/" + target_map + ".txt")
    current_map = np.zeros((map_lim, map_lim))
    for r in range(map_lim):
        for c in range(map_lim):
            if lvl[r][c] == '-':
                current_map[r][c] = 0
            elif lvl[r][c] == 'W':
                current_map[r][c] = 1
            elif lvl[r][c] == 'X':
                current_map[r][c] = 2

    target_map_list.append(current_map)
    N_obstacle = np.sum(target_map_list[0] == 1)
    N_prize = np.sum(target_map_list[0] == 2) * 3
    N = int(np.round(N_obstacle*0.66) + N_prize)

    max_possible_reward = N_prize - agent_step_cost * 100

    for iteration in range(N_iteration): # How many times this algorithm will work
        start = time.time()
        bounds = np.array([[0, map_lim**2 - 1]]*N)
        # bounds = np.array([[-map_lim, map_lim], [-map_lim, map_lim]])
        lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

        mean = lower_bounds + (np.random.rand(N) * (upper_bounds - lower_bounds))
        sigma = map_lim**2 / 4 #upper_bounds / 4.0 
        optimizer = SepCMA(mean=mean, sigma=sigma, bounds=bounds, population_size=N_population, seed=seed)

        print (f"Iteration: {iteration} - CMA Train \n")
        best_reward_list = []
        best_init_list = []
        best_pop_index_list = []

        for generation in range(N_generation):
            solutions = []
            reward_dict = dict()
            init_dict = dict()

            current_folder = output_folder + "/" +  folder_postfix + "_index_" + str(generation)
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)

            print ("Generation: ", generation)

            for pop_ind in range(optimizer.population_size):
                # start = time.time()
                map_list = []
                x = optimizer.ask()
                indices = np.clip(np.round(x), 1, map_lim**2 - 1).astype(int)
                prize_indices = indices[0:N_prize]
                obstacle_indices = indices[N_prize:]
                print ("prize_indices= ", prize_indices)
                print ("obstacle_indices= ", obstacle_indices)
                generated_map = get_map(prize_indices, obstacle_indices, map_lim)
                map_list.append(generated_map)

                train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim, visualization=visualize), n_envs=n_procs, monitor_dir=current_folder, vec_env_cls=SubprocVecEnv)
                # eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=map_lim, max_steps=20*map_lim), n_envs=n_procs, vec_env_cls=SubprocVecEnv)
                eval_env = make_vec_env(lambda: AgentFormation(generated_map=target_map_list, map_lim=map_lim, max_steps=20*map_lim), n_envs=n_procs, vec_env_cls=SubprocVecEnv)

                train_env.reset()
                eval_env.reset()

                if generation > 0:
                    model = algorithm.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation - 1) + "_best" + "/best_model", verbose=0, only_weights = False) # + "/best_model"
                    model.tensorboard_log ="./" + current_folder + "_tensorboard/"
                    model.set_env(train_env)
                    print ("Best model in gen #", (generation - 1), " is uploaded!")
                else:
                    model = algorithm('MlpPolicy', train_env, tensorboard_log="./" + current_folder + "_tensorboard/")
                
                stop_callback = StopTrainingOnNoRemarkableImprovement(max_no_improvement_evals = 100, check_percentage=0.9, verbose = 1)
                

                callback = EvalCallback(eval_env=eval_env, callback_after_eval=stop_callback, n_eval_episodes = N_eval_episodes, eval_freq = N_eval_freq, log_path = current_folder +"_log",
                                        best_model_save_path = output_folder + "/saved_models/" + folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind), deterministic=deterministic, verbose=1)
                
                model.learn(total_timesteps=train_timesteps, tb_log_name = folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind), callback=callback)
                # elapsed_time = time.time() - start
                # print (f"Elapsed time: {elapsed_time:.5} s.")        
                train_env.close()


                # Load the model
                model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_pop_" + str(pop_ind) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
                eval_env.reset()
                episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=N_eval_episodes, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                mean_reward_current = np.mean(episode_rewards)
                std_reward_current = np.std(episode_rewards)
                print (f"Mean reward in eval env: {mean_reward_current:.4f} Std reward in current env: {std_reward_current:.4f}")
                
                # target_env.reset()
                # episode_rewards, episode_lengths = evaluate_policy(model, target_env, n_eval_episodes=50, render=False, deterministic=deterministic, return_episode_rewards=True, warn=True)
                # mean_reward_target = np.mean(episode_rewards)
                # std_reward_target = np.std(episode_rewards)
                # print (f"Mean reward in target env: {mean_reward_target:.4f} Std reward in target env: {std_reward_target:.4f} ")

                eval_env.close()
                # target_env.close()

                reward_dict[pop_ind] = mean_reward_current
                init_dict[pop_ind] = indices

                current_cost = (mean_reward_current - N_prize) ** 2
                solutions.append((x, current_cost))
                print(f"#{generation} Reward: {mean_reward_current} (x1={x[0]}, x2 = {x[1]}, x3={x[2]}, x4 = {x[3]}, x5={x[4]}, x6 = {x[5]}) \n")
                # print(f"#{generation} Reward: {mean_reward_target} (x1={x[0]}, x2 = {x[1]}) ")

            optimizer.tell(solutions)

            reward_sorted = sorted(reward_dict.items(), key=operator.itemgetter(1), reverse=True)
            best_index = reward_sorted[0][0]
            best_reward_list.append(reward_sorted[0][1])
            best_init_list.append(init_dict[best_index])
            best_pop_index_list.append(best_index)

            model = model.load(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_pop_" + str(best_index) + "/best_model", verbose=1, only_weights = False) # + "/best_model"
            model.save(path=output_folder + "/saved_models/" +  folder_postfix + "_index_" + str(generation) + "_best/best_model")

            with open(folder_postfix + "_curriculum_iter_" + str(iteration) + ".pickle", 'wb') as handle:
                pickle.dump([best_reward_list, best_init_list, best_pop_index_list], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            if optimizer.should_stop():
                break
                # popsize multiplied by 2 (or 3) before each restart.
                # popsize = optimizer.population_size * 2
                # mean = lower_bounds + (np.random.rand(6) * (upper_bounds - lower_bounds))
                # optimizer = CMA(mean=mean, sigma=sigma, population_size=N_population)
                # print(f"Restart CMA-ES with popsize={N_population}")
                
        # elapsed_time = time.time() - start
        # print (f"Iteration: {iteration} Elapsed time: {elapsed_time:.5} s.")  


def curriculum_train(args):
    np.random.seed(args.seed)
    rng = default_rng(args.seed)
    agent_step_cost = 0.01

    
    # curriculum_list = ["easy", "medium", "target"]
    curriculum_list = ["level1","level2", "level3", "level4", "level5"]
    # curiculum_map_sizes = [10, 20, 30, 40]
    
    model_name = "PPO"
    model_def = PPO
    N_eval = 1000

    activation_list = [nn.Tanh]
    gamma_list = [0.9]
    bs_list = [64]
    lr_list = [3e-4]
    net_list = [[64, 64]]
    ns_list = [2048]
    ne_list = [10]
    policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])

    with open(args.map_file + '.pickle', 'rb') as handle:
        generated_map_list = pickle.load(handle)
    


    model_dir = args.out + "/saved_models"

    n_process = args.n_procs
    target_map_lim = args.map_lim


    for map_ind in range(args.n_maps):
        print ("\nCurrent map index: ", map_ind)

        if args.nocurriculum:
            map_list = []
            print ("No Curriculum")
            level = curriculum_list[-1]
            map_list.append(generated_map_list[level][map_ind])
            
            # target_map = generated_map_list[2][map_ind]
            N_reward = len(np.argwhere(map_list[0] == 2))
            max_possible_reward = N_reward - agent_step_cost * 100
            print ("max_possible_reward: ", max_possible_reward)
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_possible_reward, verbose=1)

            current_map_size = args.map_lim
            target_map_lim = args.map_lim

            train_folder = args.out + "/train_" + level + "_map_" + str(map_ind)
            if not os.path.exists(train_folder):
                os.makedirs(train_folder)

            
            val_folder = args.out + "/val_" + level + "_map_" + str(map_ind)
            if not os.path.exists(val_folder):
                os.makedirs(val_folder)

            # train_env = DummyVecEnv([lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize)])
            train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=args.map_lim, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*args.map_lim, visualization=args.visualize), n_envs=n_process, monitor_dir=train_folder, vec_env_cls=SubprocVecEnv)
            # train_env = AgentFormation(generated_map=gen_map, map_lim=args.map_lim, max_steps=250)
            # train_env = Monitor(train_env, train_folder + "/monitor.csv")
            # train_env = VecNormalize(train_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)
            train_env.reset()
            # model = model_def('MlpPolicy', train_env, policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            policy_kwargs = dict(net_arch=net_list[0], activation_fn=activation_list[0])
            model = model_def('MlpPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                            n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            
            # policy_kwargs = dict(features_extractor_class=CustomCNN, net_arch=net_list[0], features_extractor_kwargs=dict(features_dim=net_list[0][0]), activation_fn=activation_list[0])
            # model = model_def('CnnPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                        # n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")
            
            eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=val_folder, vec_env_cls=SubprocVecEnv)
            # eval_env = AgentFormation(generated_map=target_map, map_lim=args.map_lim, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*args.map_lim)
            # eval_env = VecNormalize(eval_env, norm_obs= False, norm_reward=True, clip_reward = max_possible_reward)

            callback = EvalCallback(eval_env=eval_env, callback_on_new_best=stop_callback,  eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + "_map_" + str(map_ind) +"_log",
                                    best_model_save_path = model_dir + "/best_model_" + level + "_map_" + str(map_ind), deterministic=False, verbose=1)

            start = time.time()
            # nocurriculum_train_steps = int(len(curriculum_list)*args.train_timesteps)
            model.learn(total_timesteps=args.train_timesteps, tb_log_name=model_name + "_run_" + level, callback=callback)
            model.save(model_dir + "/last_model_" + level + "_map_" + str(map_ind))
            elapsed_time = time.time() - start
            print (f"Elapsed time: {elapsed_time:.5}")

        else:
            env_list = []
            eval_list = []
            max_reward_list = []
            for index, level in enumerate(curriculum_list):
                map_list = [] # each map is given to the environment as a list containing a single map
                # current_map_size = curiculum_map_sizes[index]
                current_map_size = args.map_lim
                map_list.append(generated_map_list[level][map_ind])
                N_reward = len(np.argwhere(map_list[0] == 2))
                max_possible_reward = N_reward - agent_step_cost * 100
                max_reward_list.append(max_possible_reward)

                train_folder = args.out + "/train_" + level + "_map_" + str(map_ind)
                if not os.path.exists(train_folder):
                    os.makedirs(train_folder)

                val_folder = args.out + "/val_" + level + "_map_" + str(map_ind)
                if not os.path.exists(val_folder):
                    os.makedirs(val_folder)
                
                # train_env = DummyVecEnv([lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize)])
                # train_env = Monitor(train_env, train_folder + "/monitor.csv")
                # train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False, clip_obs=1.)
                train_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size, visualization=args.visualize), n_envs=n_process, monitor_dir=train_folder, vec_env_cls=SubprocVecEnv)
                env_list.append(train_env)

                eval_env = make_vec_env(lambda: AgentFormation(generated_map=map_list, map_lim=current_map_size, target_lim=target_map_lim, max_reward=max_possible_reward, max_steps=20*current_map_size), n_envs=n_process, monitor_dir=val_folder, vec_env_cls=SubprocVecEnv)
                # eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False, clip_obs=1.)
                eval_list.append(eval_env)

            
            for index, level in enumerate(curriculum_list):
                print (f"\nCurriculum Level: {level}")
                print ("max_possible_reward: ", max_reward_list[index])
                stop_callback = StopTrainingOnRewardThreshold(reward_threshold = max_reward_list[index], verbose=1)
                env_list[index].reset()
                previous_index = np.clip(index-1, 0, len(curriculum_list) - 1)

                model = model_def('MlpPolicy', env_list[index], n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                                    n_steps = ns_list[0],  policy_kwargs=policy_kwargs)

                # model_name = "best_model_" + curriculum_list[previous_index] + str(map_ind)
                # if os.path.exists(model_dir + "/" +  model_name + "/best_model.zip"):
                #     model = model_def.load(model_dir + "/" +  model_name + "/best_model", verbose=1)
                # else:
                #     model = model_def('MlpPolicy', env_list[index], n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                #                         n_steps = ns_list[0],  policy_kwargs=policy_kwargs)

                # policy_kwargs = dict(features_extractor_class=CNN_Network,activation_fn=activation_list[0], net_arch=net_list[0], 
                                    # features_extractor_kwargs=dict(features_dim=128))
                # model = model_def('CnnPolicy', train_env, n_epochs=ne_list[0],gamma=gamma_list[0], batch_size=bs_list[0], learning_rate=lr_list[0],
                #             n_steps = ns_list[0],  policy_kwargs=policy_kwargs, tensorboard_log="./" + args.out + "/" + model_name + "_tensorboard/")

                # model.set_env(env_list[index])
                
                callback = EvalCallback(eval_env=eval_list[index], callback_on_new_best=stop_callback, eval_freq = N_eval // n_process,
                                        best_model_save_path = model_dir + "/best_model_" + level + "_map_" + str(map_ind), deterministic=False, verbose=1)

                # if current level is target, do not appy stop callback
                # if level == curriculum_list[-1]: 
                #     callback = EvalCallback(eval_env=eval_list[index], eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                #                         best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=1)
                # else:
                #     callback = EvalCallback(eval_env=eval_list[index], callback_on_new_best=stop_callback, eval_freq = N_eval // n_process, log_path  = args.out + "/" + model_name + "_" + level + str(map_ind) +"_log",
                #                         best_model_save_path = model_dir + "/best_model_" + level + str(map_ind), deterministic=False, verbose=1)

                
                
                start = time.time()
                model.learn(total_timesteps=args.train_timesteps, callback=callback)
                model.save(model_dir + "/last_model_" + level + "_map_" + str(map_ind))
                # stats_path = os.path.join(model_dir, "vec_normalize.pkl")
                # env_list[index].save(stats_path)
                # eval_reward, _  = evaluate_policy(model, eval_list[index], n_eval_episodes=args.eval_episodes)
                elapsed_time = time.time() - start
                print (f"Elapsed time: {elapsed_time:.5}")
            

        train_env.close()