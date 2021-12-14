import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3 import A2C
import glob
from numpy.random import default_rng
import os
from train import *


def save_figs(train_str = "output_cur_00", index=0, curriculum=True):
    directory = "saved_figs/" + train_str
    mode_list = ["easy", "medium", "target"]
    color_list = ['b', 'c', 'g']
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig = plt.figure()
    if curriculum:
        pd_frame0 = load_results(train_str + "/model_outputs_" + "easy" + str(index))
        pd_frame1 = load_results(train_str + "/model_outputs_" + "medium" + str(index))
        pd_frame2 = load_results(train_str + "/model_outputs_" + "target" + str(index))

        pd_frame1['index'] += np.max(pd_frame0['index'].values)
        pd_frame2['index'] += np.max(pd_frame1['index'].values)

        sns.lineplot(x='index', y='r', data=pd_frame0, color='b')
        sns.lineplot(x='index', y='r', data=pd_frame1, color='c')
        sns.lineplot(x='index', y='r', data=pd_frame2, color='g')
        plt.legend(labels=['Easy', 'Medium', 'Target'])
        plt.title('Curriculum Learning Curve for the Map ' + str(index))
    else:
        pd_frame = load_results(train_str + "/model_outputs_" + "target" + str(index))
        sns.lineplot(x='index', y='r', data=pd_frame, color='g')
        plt.legend(labels=['Target'])
        plt.title('No-Curriculum Learning Curve for the Map ' + str(index))

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    
    plt.savefig(directory + '/map_' +str(index) + '.png')


def show_both_approaches(curriculum_folder = "output_cur_10", nocurriculum_folder = "output_nocur_10",  index=0):
    cur_directory = "saved_figs/" + curriculum_folder
    if not os.path.exists(cur_directory):
        os.makedirs(cur_directory)
    

    fig = plt.figure()
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)
    pd_frame0 = load_results(curriculum_folder + "/model_outputs_" + "easy" + str(index))
    pd_frame1 = load_results(curriculum_folder + "/model_outputs_" + "medium" + str(index))
    pd_frame2 = load_results(curriculum_folder + "/model_outputs_" + "target" + str(index))

    pd_frame1['index'] += np.max(pd_frame0['index'].values)
    pd_frame2['index'] += np.max(pd_frame1['index'].values)

    sns.lineplot(x='index', y='r', data=pd_frame0, color='b')
    sns.lineplot(x='index', y='r', data=pd_frame1, color='c')
    sns.lineplot(x='index', y='r', data=pd_frame2, color='g')
    plt.legend(labels=['Easy', 'Medium', 'Target'])
    plt.title('Curriculum Learning Curve for the Map ' + str(index))
    plt.xlabel('Episode')
    plt.ylabel('Reward')


    fig.add_subplot(1, 2, 2)
    pd_frame = load_results(nocurriculum_folder + "/model_outputs_" + "target" + str(index))
    sns.lineplot(x='index', y='r', data=pd_frame, color='g')
    plt.legend(labels=['Target'])
    plt.title('No-Curriculum Learning Curve for the Map ' + str(index))

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    
    plt.savefig(cur_directory + '/map_' +str(index) + '.png')
    plt.show()


def create_maps(seed = 7, map_lim=30, N_maps = 10):
    np.random.seed(seed)
    rng = default_rng(seed)
    gen_list, easy_list, medium_list = [], [], []
    for i in range(N_maps):
        gen_map = generate_maps(map_lim=map_lim)
        easy_map = curriculum_design(gen_map, rng, level = "easy")
        medium_map = curriculum_design(gen_map, rng, level = "medium")
        gen_list.append(gen_map)
        easy_list.append(easy_map)
        medium_list.append(medium_map)
        
    with open('saved_maps_' + str(map_lim) + '.pickle', 'wb') as handle:
        pickle.dump([easy_list, medium_list, gen_list], handle, protocol=pickle.HIGHEST_PROTOCOL)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(folder="output_nocur", index=0, title='No-Curriculum Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    current_folder = folder + "/model_outputs_target" + str(index)
    pd_frame = load_results(current_folder).values
    x, y = pd_frame[:,0], pd_frame[:,1]
    # x, y = ts2xy(load_results(current_folder), 'timesteps')
#     y = moving_average(y, window=50)
#     # Truncate x
#     x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Rewards')
    plt.title(title + " for the Map " + str(index))
    plt.show()

def plot_curriculum_results(folder="output_cur", map_lim=20000, index=0, title='Curriculum Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x_list, y_list = [], []
    x_new, y_new = [], []
    xposition = [0, 0, 0, 0]
    x_offset = [0, map_lim, 2*map_lim]
    level_list = ["easy", "medium", "target"]
    folder_prefix = folder + "/model_outputs_"
    for i, level in enumerate(level_list): 
        folder_name = folder_prefix + level + str(index)
        pd_frame = load_results(folder_name).values
        x, y = pd_frame[:,0], pd_frame[:,1]
        # x, y = ts2xy(load_results(folder_name), 'timesteps')
        if len(x) == 0:
            continue
        xposition[i+1] = x[-1] + xposition[i]
        y_list.append(y)
        x_list.append(x)
        

    for i in range(len(y_list)):
        for j in range(len(y_list[i])):
            y_new.append(y_list[i][j])
            x_new.append(x_list[i][j] + xposition[i])
            
    y_new = np.asarray(y_new)
    x_new = np.asarray(x_new)
        
    fig = plt.figure(title)
    plt.plot(x_new, y_new)
    colors = ['r','k']
    labels = ["easy|medium", "medium|target"]
    for j, xc in enumerate(xposition[1:-1]):
        plt.axvline(x=xc,linestyle='--', label='{}'.format(labels[j]), c=colors[j])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Rewards')
    plt.title(title + " for the Map " + str(index))
    plt.legend(),
    plt.show()


def plot_both_approaches(folder="output_cur", index=0, title='Curriculum Learning Curve', folder2="output_nocur", title2='No-Curriculum Learning Curve', limit=30000):
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 2, 1)

    x_list, y_list = [], []
    x_new, y_new = [], []
    xposition = [0, 0, 0, 0]
    x_offset = [0, limit, 2*limit]
    level_list = ["easy", "medium", "target"]
    folder_prefix = folder + "/model_outputs_"
    for i, level in enumerate(level_list): 
        folder_name = folder_prefix + level + str(index)
        x, y = ts2xy(load_results(folder_name), 'timesteps')
        if len(x) == 0:
            continue
        xposition[i+1] = x[-1] + x_offset[i]
        y_list.append(y)
        x_list.append(x)
        

    for i in range(len(y_list)):
        for j in range(len(y_list[i])):
            y_new.append(y_list[i][j])
            x_new.append(x_list[i][j] + x_offset[i])
            
    y_new = np.asarray(y_new)
    x_new = np.asarray(x_new)
        
    # fig = plt.figure(title)
    plt.plot(x_new, y_new)
    colors = ['r','k']
    labels = ["easy|medium", "medium|target"]
    for j, xc in enumerate(xposition[1:-1]):
        plt.axvline(x=xc,linestyle='--', label='{}'.format(labels[j]), c=colors[j])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " for the Map " + str(index))
    plt.legend()

    fig.add_subplot(1, 2, 2)


    current_folder = folder2 + "/model_outputs_target" + str(index)
    x, y = ts2xy(load_results(current_folder), 'timesteps')

    # fig = plt.figure(title2)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title2 + " for the Map " + str(index))
    
    


    plt.show()