import numpy as np
from dstar import *

#Construct map from file
def ff_regenate(file_name):
    #for file in x:
    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    txt_data = np.loadtxt(file_name, dtype=str)
    ds_map = Map(len(txt_data[:]), len(txt_data[1]))

    map_lim = len(txt_data[:])

    matrix_map = np.zeros((3,len(txt_data[:]), len(txt_data[1])))

    for x in range(len(txt_data[:])):
        for y in range(len(txt_data[1])):
            if txt_data[x][y] == 'W':
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([y, x])
                matrix_map[1,x,y] = 1
            elif txt_data[x][y] == 'X':
                prize_locations.append([y, x])
                matrix_map[2,x,y] = 1
            else: #if txt_data[x][y] == '-':
                matrix_map[0,x,y] = 1

    #print("matrix_map: ", matrix_map.shape)
    ds_map.set_obstacle([(i, j) for i, j in zip(obs_x_list, obs_y_list)])
    #print("file_name: ", str(file_name), " prize_locations : ", prize_locations)
    return ds_map, obstacle_locations, prize_locations, matrix_map, map_lim

#Construct map from text array
def fa_regenate(array):
    array = [item.replace("\n", "") for item in array]

    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    ds_map = Map(len(array), len(array[0]))
    map_lim = len(array[0])

    matrix_map = np.zeros((3,len(array), len(array[0])))
    
    for x in range(len(array[0])): #row
        for y in range(len(array[1])): #column
            if array[x][y] == 'W':
                if(1<= x <=4 and 1<= y <=4):
                    continue
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([x, y])
                matrix_map[1,x,y] = 1
            elif array[x][y] == 'X':
                if(1<= x <=4 and 1<= y <=4):
                    continue
                prize_locations.append([x, y])
                matrix_map[2,x,y] = 1
            else: #if array[x][y] == '-':
                matrix_map[0,x,y] = 1

    ds_map.set_obstacle([(i, j) for i, j in zip(obs_y_list, obs_x_list)])
    return ds_map, obstacle_locations, prize_locations, matrix_map, map_lim, obs_y_list, obs_x_list

def fa_regenate2(matrix_map):
    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    ds_map = Map(40, 40)
    map_lim = 40

    for x in range(40): #row
        for y in range(40): #column
            if matrix_map[1,x,y] == 1:
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([x, y])
            if matrix_map[2,x,y] == 1:
                prize_locations.append([x, y])

    ds_map.set_obstacle([(i, j) for i, j in zip(obs_y_list, obs_x_list)])
    return ds_map, obstacle_locations, prize_locations, matrix_map, map_lim, obs_y_list, obs_x_list

#Construct map from numpy array
def fa_convert(matrix_map):

    prize_locations = []
    obstacle_locations = []
    obs_x_list = []
    obs_y_list = []
    ds_map = Map(len(matrix_map), len(matrix_map[0]))
    map_lim = len(matrix_map[0])


    for x in range(len(matrix_map[0])): #row
        for y in range(len(matrix_map[1])): #column
            if matrix_map[1][x][y] == 1:
                obs_x_list.append(x)
                obs_y_list.append(y)
                obstacle_locations.append([x, y])
            elif matrix_map[2][x][y] == 1:
                prize_locations.append([x, y])

    #print("matrix_map: ", matrix_map)
    ds_map.set_obstacle([(i, j) for i, j in zip(obs_y_list, obs_x_list)])
    # print("gelen matrix_map: ", matrix_map)
    # print("obstacle_locations: ", obstacle_locations)
    # print("prize_locations: ", prize_locations)
    return ds_map, obstacle_locations, prize_locations, matrix_map, map_lim, obs_y_list, obs_x_list



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