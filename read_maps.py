import numpy as np
import glob
import os
from dstar import *

def read_txt(file_name):
    #for file in x:
    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    txt_data = np.loadtxt(file_name, dtype=str)
    ds_map = Map(len(txt_data[:]), len(txt_data[1]))

    for x in range(len(txt_data[:])):
        for y in range(len(txt_data[1])):
            if txt_data[x][y] == 'O' or txt_data[x][y] == 'W':
                obs_x_list.append(x)
                obs_y_list.append(y)
                #current_pos = [y, x]
                #if current_pos not in obstacle_locations:
                obstacle_locations.append([y, x])
            elif txt_data[x][y] == 'X':
                prize_locations.append([y, x])

    
    ds_map.set_obstacle([(i, j) for i, j in zip(obs_x_list, obs_y_list)])
    #print("file_name: ", str(file_name), " obstacle_locations : ", obstacle_locations)
    return ds_map, obstacle_locations, prize_locations

if __name__ == "__main__":
    directory = './output/wandb/latest-run/files/random_samples/txt/'
    dir_names = os.listdir(directory)
    dir_names.sort()

    x = sorted(glob.glob(directory + "*.txt"))
    print("x: ", x[1])
    ds_map, obstacle_locations, prize_locations = read_txt(x[1])
    print("prize_locations: ", prize_locations)