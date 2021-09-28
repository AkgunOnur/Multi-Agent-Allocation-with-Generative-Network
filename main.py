import wandb
import sys
import torch
import os
import pandas as pd

from loguru import logger
from classifier import LeNet
from train import GAN
from read_maps import *
from torch.optim import Adam

from construct_library import Library
from generate_random_map import generate_random_map
from env_funcs import env_class
from config import get_arguments, post_config
from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS
from environment.level_image_gen import LevelImageGen as LevelGen
from environment.level_utils import read_level, one_hot_to_ascii_level


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]

colon = ['testc_labeled', 'train_loss', 'trainc_labeled', 'train_lib_size']

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

    L = Library(180)
    e = env_class()

    L.add(read_level(opt, None, replace_tokens).to(opt.device),5)#6 agent

    classifier = LeNet(numChannels=3, classes=6).to(opt.device) #(0-5) = 6 is max agent number in map

    optimizer = Adam(classifier.parameters(), lr=1e-4)
    
    classifier.eval()
    testc_labeled = classifier.predict(L.test_library)
    write_tocsv([testc_labeled, 0.0, 0,  0])
    
    print("MODE:", opt.mode)
    if(opt.mode == 'train'):

        g = GAN(opt)

        for s in range(180):
            classifier.load_state_dict(torch.load("./classifier_init.pth"))

            classifier.train()
            training_loss, trainc_labeled = classifier.trainer(L.train_library, optimizer)

            classifier.eval()
            testc_labeled = classifier.predict(L.test_library)
            
            #Log Data
            write_tocsv([testc_labeled, training_loss, trainc_labeled, s])

            while(True):
                sample_map, _ = L.get()

                #Train GAN and return fake map
                generated_map = g.train(e, sample_map, classifier, opt)
                coded_fake_map = one_hot_to_ascii_level(generated_map.detach(), opt.token_list)
                ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

                #Decide whether place the generated map in the training lib
                classifier.eval()
                prediction =  classifier.predict2(torch.from_numpy((agent_map.reshape(1,3,40,40))).float())

                # run D* for all possible n_agents and find best
                rewards = []
                for i in range(6):
                    reward = e.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                    rewards.append(reward)
                #Get actual best n_agents

                actual = np.argmax(rewards)

                #Decide whether place the generated map in the training lib
                if(prediction==actual): #no need to add library
                  continue
                else:
                  L.add(agent_map, prediction) #add it to training library
                  if (s%10==0 and s>0):
                    g.better_save(s)
                  break

    elif(opt.mode == 'random_without_gan'):
        for s in range(180):
            #Reset classifier
            classifier.load_state_dict(torch.load("./classifier_init.pth"))

            #train classifier with training library
            classifier.train()
            training_loss, trainc_labeled = classifier.trainer(L.train_library, optimizer)

            #Test classifier perf on test library
            classifier.eval()
            testc_labeled = classifier.predict(L.test_library)

            #Log Data
            # write_tocsv([testc_labeled, training_loss, trainc_labeled, s], file_name ='rwg_performance.csv')

            # while condition to repeat training until training lib expand
            while(True):
                #Generate Random Map and add it to training library
                generated_map = generate_random_map(40)
                ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate2(generated_map)

                #Decide whether place the generated map in the training lib
                classifier.eval()
                prediction =  classifier.predict2(torch.from_numpy((agent_map.reshape(1,3,40,40))).float())

                rewards = []
                for i in range(6):
                    reward = e.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                    rewards.append(reward)
                #Get actual best n_agents
                actual = np.argmax(rewards)


                #Decide whether place the generated map in the training lib
                if(prediction==actual): #no need to add library
                  continue
                else:
                  L.add(agent_map, prediction) #add it to training library
                  break
        os.rename('./training_map_library.pkl', './training_maps_random_without_gan.pkl')

    elif(opt.mode == 'random_train'):
        for s in range(180):
            #Generate Random Map and add it to training library
            generated_map = generate_random_map(40)
            ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate2(generated_map)

            rewards = []
            for i in range(6):
                reward = e.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                rewards.append(reward)
            #Get actual best n_agents
            actual = np.argmax(rewards)
            L.add(agent_map,actual)
        
        #Reset classifier
        classifier.load_state_dict(torch.load("./classifier_init.pth"))

        #train classifier with training library
        classifier.train()
        training_loss, trainc_labeled = classifier.trainer(L.train_library, optimizer)

        #Test trained classifier perf on test library and log perf.
        classifier.eval()
        testc_labeled = classifier.predict(L.test_library)
        print("testc_labeled:", testc_labeled, "training_loss:", training_loss, "trainc_labeled:", trainc_labeled,  "s:", s)
        write_tocsv([testc_labeled, training_loss, trainc_labeled,  s],file_name='random_performance.csv')
        os.rename('./training_map_library.pkl', './training_maps_random.pkl')
    else:
        print("Unnoticeable Working Mode")

if __name__ == "__main__":
    main()
