import random
import numpy as np
import os
import pickle
from read_maps import *
from env_funcs import env_class
from config import get_arguments, post_config
from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS
from environment.level_image_gen import LevelImageGen as LevelGen
from environment.level_utils import read_level, one_hot_to_ascii_level


# This function return matrix type map for obstacles and prizes
def generate_random_map(map_size):
    matrix_map = np.zeros((3,map_size,map_size))

    N_prize = (map_size/20)*random.randint(3,15)
    N_obstacle = (map_size/20)*random.randint(40,map_size*5)

    prize_locations = []
    obstacle_locations = []

    #placing ground in matrix map
    for x in range(1,map_size-1):
        for y in range(1,map_size-1):
                matrix_map[0,x,y] = 1

    while(len(prize_locations)<N_prize):
        x = random.randint(1,map_size-2)
        y = random.randint(1,map_size-2)
        if([x,y] not in prize_locations and (x>=4 or y>=4)):
            prize_locations.append([x,y])
            matrix_map[2,x,y] = 1
            matrix_map[0,x,y] = 0


    while(len(obstacle_locations)<N_obstacle):
        x = random.randint(0,map_size-1)
        y = random.randint(0,map_size-1)
        if([x,y] not in (obstacle_locations and prize_locations) and (x>=4 or y>=4)):
            obstacle_locations.append([x,y])
            matrix_map[1,x,y] = 1
            matrix_map[0,x,y] = 0

    #Draw border around map
    matrix_map[1,0,:] = 1
    matrix_map[1,:,0] = 1
    matrix_map[1,map_size-1,:] = 1
    matrix_map[1,:,map_size-1] = 1
    #print("matrix_map: ", matrix_map)
    return matrix_map


# This function creates test maps in matrix form and saves to folder in txt format
def construct_test_maps(N_maps, map_dir='./test_maps', map_size = 20):
    env = env_class()
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

    test_library = [[],[]]
    for n in range(N_maps):
        matrix_map = generate_random_map(map_size)
        #Find label of the map
        ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate2(matrix_map)
        rewards = []
        for i in range(3):
            reward = env.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
            rewards.append(reward)
        
        label = np.argmax(rewards)
        #Add map to pickle
        test_library[0].append(np.array(matrix_map).reshape((3,20,20)))
        test_library[1].append(np.array(label))
        with open(f'./test_maps/test_map_library.pkl', 'wb') as f:
            pickle.dump(test_library, f)

        #Return to txt file
        ascii_level = []
        for i in range(matrix_map.shape[1]):
            line = ""
            for j in range(matrix_map.shape[2]):
                if(i ==0 or i == matrix_map.shape[2] - 1):
                    line += 'W'
                elif(j ==0 or j == matrix_map.shape[2] - 1):
                    line += 'W'
                elif(matrix_map[0,i,j]==1):
                    line += '-'
                elif(matrix_map[1,i,j]==1):
                    line += 'W'
                elif(matrix_map[2,i,j]==1):
                    line += 'X'
                else:
                    print("Error!!!")
            ascii_level.append(line)

        #Write to txt file
        with open(os.path.join(map_dir,'test'+str(n+1)+'.txt'), "w") as tf:
            for element in ascii_level:
                tf.write(element + "\n")

if __name__ == "__main__":
    construct_test_maps(50)