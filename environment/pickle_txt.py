# Code inspired by https://github.com/tamarott/SinGAN


from tokens import REPLACE_TOKENS as REPLACE_TOKENS #environment.tokens

import numpy as np
import os
import pickle


class TXT_Library():
    #Initialize library
    def __init__(self):

        target = f'/home/avsp/Masaüstü/ganner/training_libs/training_map_random.pickle'
        
        if os.path.getsize(target) > 0:      
            with open(target, "rb") as f:
                unpickler = pickle.Unpickler(f)
                self.train_library = unpickler.load()

def write_maps(map, index):
    
    txt_list = []
    for x in range(map.shape[1]): #row
        line = []
        for y in range(map.shape[2]): #column
            if map[0][x][y] == 1.0:
                line.append('-')
            elif map[1][x][y] == 1.0:
                line.append('W')
            elif map[2][x][y] == 1.0:
                line.append('X')
            else:
                line.append('-')
        lin = ''.join(line)
        txt_list.append(lin)

    with open(f"/home/avsp/Masaüstü/ganner/output/random_maps/map-{index+1}.txt", "w+", encoding="latin-1") as output:
        for lne in txt_list:
            output.write(str(lne))
            output.write('\n')
        
def write():
    #Initialize Library and environment
    L = TXT_Library()
    for idx in range(len(L.train_library)):
        write_maps(np.asarray(L.train_library[idx]), idx)


if __name__ == "__main__":
    write()