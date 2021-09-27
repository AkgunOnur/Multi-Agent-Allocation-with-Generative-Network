# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
#from test import test

from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS

from environment.level_image_gen import LevelImageGen as LevelGen
from environment.level_utils import read_level, one_hot_to_ascii_level

from config import get_arguments, post_config
from loguru import logger
import wandb
import sys
import torch

from construct_library import Library
from generate_random_map import generate_random_map
from env_funcs import env_class
from classifier import LeNet
from train import GAN
from read_maps import *
import pandas as pd
import os
import torch
from torch.optim import Adam
from argparse import ArgumentTypeError
import numpy as np
import os
import pickle
from generate_random_map import generate_random_map

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
            target = './training_maps_random_without_gan.pkl'
        elif(mode=='test_random'):
            target = './training_maps_random.pkl'
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
        
def main():
    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    #Initialize Library and environment
    L = Library(opt.testmode)

    #Initalize classifier and load its weights
    classifier = LeNet(numChannels=3, classes=6).to(opt.device) #(0-5) = 6 is max agent number in map
    classifier.load_state_dict(torch.load("./classifier_init.pth"))

    # initialize classifier optimizer and loss function
    optimizer = Adam(classifier.parameters(), lr=1e-4)

    #train classifier with training library
    classifier.train()
    training_loss, trainc_labeled = classifier.trainer(L.train_library, optimizer)

    #Test classifier perf on test library
    classifier.eval()
    testc_labeled = classifier.predict(L.test_library)
    
    print("training_loss: ", training_loss, "trainc_labeled: ", trainc_labeled, "testc_labeled: ", testc_labeled)

if __name__ == "__main__":
    main()
