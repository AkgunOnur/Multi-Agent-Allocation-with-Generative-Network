import os
import sys
import glob
import pickle
import numpy as np
from pathlib import Path
from level_utils import load_level_from_text
from level_image_gen import LevelImageGen

# Renders all level.txt files to png images inside a given folder. Expects ONLY .txt in that folder.

def get_indices(numbers, map_lim):
    index = []
    for number in numbers:
        x = number // map_lim
        y = number % map_lim
        index.append([x,y])
    return np.array(index)

def create_map_files(map_file, level_list, map_lim = 10):
    predefined_obtacles = get_indices([i  for i in range(map_lim**2) if i% map_lim == 0 or i % map_lim == map_lim - 1 \
                                                                    or i // map_lim == 0 or i // map_lim == map_lim - 1], map_lim)
    # level_list = ["easy", "medium", "target"]
    with open('../' + map_file + '.pickle', 'rb') as handle:
        generated_map_list = pickle.load(handle)
            
    for level in level_list:
        map_list = []
        current_list = np.copy(generated_map_list[level])

        # if level == "easy":
        #     current_list = np.copy(easy_list)
        # elif level == "medium":
        #     current_list = np.copy(medium_list)
        # elif level == "target":
        #     current_list = np.copy(gen_list)
                
        for current_map in current_list:
            current_map[0][predefined_obtacles[:,0], predefined_obtacles[:,1]] = 1 # predefined walls
            current_map[0][1,1] = 0 # Agent start location, obstacle free
            map_list.append(current_map[0])


        directory = "../maps/" + map_file + "/" + level
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for ind, current_map in enumerate(map_list):
            matrix = np.asarray([['-' for i in range(map_lim)] for j in range(map_lim)])
            matrix[current_map == 1] = 'W'
            matrix[current_map == 2] = 'X'

            with open(directory + "/map_" + str(ind) + ".txt", "w") as txt_file:
                for line in matrix:
                    txt_file.write("".join(line) + "\n") # works with any number of elements in a line



if __name__ == '__main__':
    SPRITE_PATH = "./sprites"
    map_lim = 30
    map_file = "new_saved_maps_30"
    output_folder = "rendered_" + map_file
    # level_list = ["easy", "medium", "target"]
    level_list = ["level1", "level2", "level3", "level4", "level5"]

    
    create_map_files(map_file = map_file, level_list=level_list, map_lim=map_lim)
    
    ImgGen = LevelImageGen(SPRITE_PATH)
    for level in level_list:

        directory = '../maps/' + map_file + "/" + level + "/"

        map_names = glob.glob(directory + "*.txt")

        for map_name in map_names:
            name = map_name.split('\\')[-1]
            print("map_name:", name)

            target_dir = '../maps/' + output_folder + "/" + level
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)

            lvl = load_level_from_text(map_name)
            #print("lvl: ", lvl)
            if lvl[-1][-1] == '\n':
                lvl[-1] = lvl[-1][0:-1]
            lvl_img = ImgGen.render(lvl)
            lvl_img.save(os.path.join(target_dir, name[0:-4] + '.png'), format='png')
