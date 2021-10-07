# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
#from test import test

from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS

from environment.level_image_gen import LevelImageGen as LevelGen
from environment.level_utils import read_level, one_hot_to_ascii_level

from construct_library import Library
from env_funcs import env_class
from classifier import LeNet
from train import GAN
from read_maps import *
import numpy as np
import os
import pickle


class Library():
    #Initialize library
    def __init__(self,mode):
        #Load test maps
        target = './test_dataset.pickle'
        if os.path.getsize(target) > 0:      
            with open(target, "rb") as f:
                unpickler = pickle.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                self.test_library = unpickler.load()

        self.test_library[1] = [x-1 for x in self.test_library[1]]

        #Load training maps
        if(mode=='test_gan'):
            target = './training_map_library.pkl'
        elif(mode=='test_wo_gan'):
            target = './Training_Libs/training_maps_random_without_gan.pkl'
        elif(mode=='test_random'):
            target = './Training_Libs/training_maps_random.pkl'
        else:
            print("WARNING: Failed to select test maps!!!!")
        if os.path.getsize(target) > 0:      
            with open(target, "rb") as f:
                unpickler = pickle.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                self.train_library = unpickler.load()
                self.train_library[1][0] = np.array(self.train_library[1][0] - 1)
                # print("self.train_library[0]: ", self.train_library[1])
                # armut

def write_maps(map, index):
    txt_list = []
    print(map.shape)
    for x in range(map.shape[1]): #row
        line = []
        print(map[0][x])
        print(map[1][x])
        print(map[2][x])
        for y in range(map.shape[2]): #column
            if map[0][x][y] == 1:
                line.append('-')
            elif map[1][x][y] == 1:
                line.append('W')
            elif map[2][x][y] == 1:
                line.append('X')
            else:
                line.append('-')
        # line.append('\n')
        lin = ''.join(line)
        print(lin)
        # print(line)
        txt_list.append(lin)
        # print(txt_list)

    with open(f"./output/generated_maps/map-{index}.txt", "w+", encoding="latin-1") as output:
        for lne in txt_list:
            output.write(str(lne))
            output.write('\n')
        
def write():
    #Initialize Library and environment
    L = Library('test_gan')
    for idx in range(len(L.train_library[0])):
        #print("len(L.train_library): ", len(L.train_library[0]))
        write_maps(np.asarray(L.train_library[0][idx]).squeeze(axis=0), idx)


if __name__ == "__main__":
    write()