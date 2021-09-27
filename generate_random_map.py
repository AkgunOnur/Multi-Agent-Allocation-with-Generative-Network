import random
import numpy as np
import os

# This function return matrix type map for obstacles and prizes
def generate_random_map(map_size):
    matrix_map = np.zeros((3,map_size,map_size))

    N_prize = (map_size/40)*random.randint(10,30)
    N_obstacle = (map_size/40)*random.randint(40,map_size*2)

    prize_locations = []
    obstacle_locations = []

    while(len(prize_locations)<N_prize):
        x = random.randint(1,map_size-2)
        y = random.randint(1,map_size-2)
        if([x,y] not in prize_locations):
            prize_locations.append([x,y])


    while(len(obstacle_locations)<N_obstacle):
        x = random.randint(1,map_size-2)
        y = random.randint(1,map_size-2)
        if([x,y] not in obstacle_locations and prize_locations):
            obstacle_locations.append([x,y])
    

    #placing prizes in matrix map
    for i in range(int(N_prize)):
        [x,y] = prize_locations[i]
        if(1<=x<=4 and 1<=y<=4):
            continue
        matrix_map[2,x,y] = 1
    
    #placing obstacles in matrix map
    for i in range(int(N_obstacle)):
        [x,y] = obstacle_locations[i]
        if(1<=x<=4 and 1<=y<=4):
            continue
        matrix_map[1,x,y] = 1
    
    #placing ground in matrix map
    for x in range(40):
        for y in range(40):
            if(matrix_map[2,x,y]==0 and matrix_map[1,x,y]==0):
                matrix_map[0,x,y] = 1

    #Draw border around map
    # matrix_map[1,0,:] = 1
    # matrix_map[1,:,0] = 1
    # matrix_map[1,map_size-1,:] = 1
    # matrix_map[1,:,map_size-1] = 1

    return matrix_map


# This function creates test maps in matrix form and saves to folder in txt format
def construct_test_maps(N_maps, map_dir='test_maps', map_size = 80, scales=[0.25, 0.5, 0.75, 1]):
    if not os.path.exists(map_dir):
        for i in range(len(scales)):
            os.makedirs(os.path.join(map_dir,str(int(map_size*scales[i]))))
    
    for s in range(len(scales)):
        for n in range(N_maps):
            matrix_map = generate_random_map(int(scales[s]*map_size))

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
                        line += 'X'
                    elif(matrix_map[1,i,j]==1):
                        line += 'W'
                    else:
                        line += '-'
                ascii_level.append(line)

            #Write to txt file
            with open(os.path.join(map_dir, str(int(scales[s]*map_size)),'test'+str(n+1)+'.txt'), "w") as tf:
                for element in ascii_level:
                    tf.write(element + "\n")
