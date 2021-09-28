# Code inspired by https://github.com/tamarott/SinGAN
from construct_library import Library
from generate_random_map import generate_random_map
from classifier import LeNet
from train import GAN
from read_maps import *
import pandas as pd
import torch
from torch.optim import Adam
import torch
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
            target = './Training_Libs/training_map_library.pkl'
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
