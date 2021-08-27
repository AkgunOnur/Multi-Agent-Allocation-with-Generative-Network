import numpy as np
import glob
import os
from dstar import *

def ff_regenate(file_name):
    #for file in x:
    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    txt_data = np.loadtxt(file_name, dtype=str)
    ds_map = Map(len(txt_data[:]), len(txt_data[1]))

    matrix_map = np.zeros((2,len(txt_data[:]), len(txt_data[1])))

    for x in range(len(txt_data[:])):
        for y in range(len(txt_data[1])):
            if txt_data[x][y] == 'O' or txt_data[x][y] == 'W':
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([y, x])
                matrix_map[1,x,y] = 1
            elif txt_data[x][y] == 'X':
                prize_locations.append([y, x])
                matrix_map[0,x,y] = 1

    #print("matrix_map: ", matrix_map.shape)
    ds_map.set_obstacle([(i, j) for i, j in zip(obs_x_list, obs_y_list)])
    print("file_name: ", str(file_name), " prize_locations : ", prize_locations)
    return ds_map, obstacle_locations, prize_locations, matrix_map

def fa_regenate(array):
    array = [item.replace("\n", "") for item in array]

    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    ds_map = Map(len(array), len(array[0]))

    matrix_map = np.zeros((2,len(array), len(array[0])))

    for x in range(len(array[0])): #row
        for y in range(len(array[1])): #column
            if array[x][y] == 'O' or array[x][y] == 'W':
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([y, x])
                matrix_map[1,x,y] = 1
            elif array[x][y] == 'X':
                prize_locations.append([y, x])
                matrix_map[0,x,y] = 1

    #print("matrix_map: ", matrix_map)
    ds_map.set_obstacle([(i, j) for i, j in zip(obs_x_list, obs_y_list)])
    #print("file_name: ", str(file_name), " prize_locations : ", prize_locations)
    return ds_map, obstacle_locations, prize_locations, matrix_map



# if __name__ == "__main__":
#     x = ['XWWOXOWXOX\n', 'OOOWXXWOXW\n', 'WOXWX-OWXX\n', 'OOOOOXOWOX\n', 'OX-XXOOO-W\n', 'OOXOOOOWXX\n', 'OXOWOOOOXW\n', 'OWOXOXWXWW\n', 'WXOWXWOXXX\n', '-XXOXOXOWW']
#     x = [item.replace("\n", "") for item in x]
#     print(x)
#     fa_regenate(x)
#     directory = './output/wandb/latest-run/files/random_samples/txt/'
#     dir_names = os.listdir(directory)
#     dir_names.sort()

#     x = sorted(glob.glob(directory + "*.txt"))
#     print("x: ", x[5])
#     ds_map, obstacle_locations, prize_locations, matrix_map = read_and_generate(x[5])
#     # print("prize_locations: ", prize_locations)