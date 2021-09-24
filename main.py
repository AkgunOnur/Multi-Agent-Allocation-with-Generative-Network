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
from env_funcs import env_class
from classifier import LeNet
from train import GAN
from read_maps import *
import pandas as pd
import os
import torch
from torch.optim import Adam

def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]

colon = ['test_loss', 'train_loss', 'train_lib_size']

def write_tocsv(stats,file_name='performance.csv'):
    df_stats = pd.DataFrame([stats], columns=colon)
    df_stats.to_csv(file_name, mode='a', index=False,header=not os.path.isfile(file_name))

def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """
    #==================================================================================
    # torch.autograd.set_detect_anomaly(True)
    # Logger init
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

    # Parse arguments
    opt = get_arguments().parse_args()
    opt = post_config(opt)

    # Init wandb
    run = wandb.init(project="environment", tags=get_tags(opt),
                     config=opt, dir=opt.out, mode="offline")
    opt.out_ = run.dir
    # Init game specific inputs
    sprite_path = opt.game + '/sprites'
    opt.ImgGen = LevelGen(sprite_path)
    replace_tokens = REPLACE_TOKENS
    #==================================================================================

    #Initialize Library and environment
    L = Library(180)
    e = env_class()

    #Add first (map,label) into the library
    L.add(read_level(opt, None, replace_tokens).to(opt.device),6)

    #Initalize classifier and save weights
    classifier = LeNet(numChannels=3, classes=6).to(opt.device) #(0-6) = 6 is max agent number in map

    # initialize classifier optimizer and loss function
    optimizer = Adam(classifier.parameters(), lr=1e-4)
    
    #==== WARNING: Dont forget to comment out next line ====#
    torch.save(classifier.state_dict(), "./classifier_init.pth")

    #Test initial classifier perf on test library and log perf.
    classifier.eval()
    test_loss = classifier.predict(L.test_library)
    write_tocsv([test_loss, 0.0, 0])

    if(opt.mode == 'train'):
        g = GAN(opt)

        for s in range(180):
            #Reset classifier
            classifier.load_state_dict(torch.load("./classifier_init.pth"))

            #train classifier with training library
            classifier.train()
            training_loss = classifier.trainer(L.train_library, optimizer)

            #Test classifier perf on test library
            classifier.eval()
            test_loss = classifier.predict(L.test_library)
            
            #Log Data
            write_tocsv([test_loss, training_loss, s])

            # while condition to repeat GAN training until training lib expand
            while(True):
                #get a random real = ([map, label]) from training library
                sample_map, _ = L.get()

                #Train GAN and return fake map
                generated_map = g.train(e, sample_map, classifier, opt)
                coded_fake_map = one_hot_to_ascii_level(generated_map.detach(), opt.token_list)
                ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

                #Decide whether place the generated map in the training lib
                classifier.eval()
                prediction =  classifier.predict2(torch.from_numpy((agent_map.reshape(1,3,40,40))).float()) + 1 
                # run D* for all possible n_agents and find best
                rewards = []
                for i in range(6):
                    reward = e.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                    rewards.append(reward)
                #Get actual best n_agents
                actual = np.argmax(rewards)+1

                #Decide whether place the generated map in the training lib
                if(prediction==actual): #no need to add library
                  continue
                else:
                  L.add(agent_map, prediction) #add it to training library
                  if (s%10==0 and s>0):
                    g.better_save(s)
                  break
        
        #TODO:Repeat same process for random training library
    #TODO: test mode
    # elif(opt.mode == 'test'):
    #     #Load model and switch eval mode
    #     # classifier.load_state_dict(torch.load("classifier.pt"))
    #     # classifier.eval()
    #     test(opt)
    else:
        print("Unnoticeable Working Mode")

if __name__ == "__main__":
    main()
