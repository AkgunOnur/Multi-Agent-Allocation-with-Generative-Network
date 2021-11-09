import wandb
import sys
import torch
import os

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

from torch.utils.tensorboard import SummaryWriter

def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]

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
    #Initialize library and environment
    gen_lib = Library(opt.library_size)
    env = env_class()

    #Initialize tensorboard logger
    writer = SummaryWriter()

    #Initialize classifier and save initial weights
    classifier = LeNet(numChannels=3, classes=3, args=opt).to(opt.device)
    #ACHTUNG!!! - >>> Comment Time to TIME when neccessary
    #torch.save(classifier.state_dict(),"./weights/classifier_init.pth")

    classifier.load_state_dict(torch.load("./weights/classifier_init.pth"))

    #Predictions with initial logs
    classifier.eval()
    #Test classifier perf on test library
    testc_labeled = classifier.predict(gen_lib.test_library)

    #logs using tensorboard
    writer.add_scalar("testc_labeled",testc_labeled,0)

    #Initialize optimizer
    optimizer = Adam(classifier.parameters(), lr=1e-4)

    ########################################################
    #Add initial maps and their labels to training library
    for lvl in range(3):
        if lvl == 0:
            opt.input_name = "easy_map.txt"
        elif lvl == 1:
            opt.input_name = "medium_map.txt"
        elif lvl == 2:
            opt.input_name = "hard_map.txt"

        #Find labels of initial maps
        init_map = read_level(opt, None, replace_tokens)
        init_ascii_map = one_hot_to_ascii_level(init_map, opt.token_list)
        ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate(init_ascii_map)
        rewards = []
        for i in range(3):
            reward = env.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
            rewards.append(reward)
        #Add best nagent as label
        gen_lib.add(init_map,np.argmax(rewards),opt)

    
    #Reset classifier weights to initial and switch to train mode
    classifier.train()

    #Train classifier with only in the training library maps
    training_loss, trainc_labeled = classifier.trainer(gen_lib.train_library[0], gen_lib.train_library[1], optimizer)

    #Test classifier perf on test library
    classifier.eval()
    testc_labeled = classifier.predict(gen_lib.test_library)

    #logs using tensorboard 
    writer.add_scalar("trainc_labeled",trainc_labeled,0)
    writer.add_scalar("testc_labeled",testc_labeled,1)
    writer.add_scalar("training_loss", training_loss, 0)
    ########################################################
    #Write selected mode
    print("MODE:", opt.mode)
    idx = 0
    
    #Train loop
    if(opt.mode == 'GAN' or opt.mode == 'gan'):
        #Initialize GAN
        g = GAN(opt)
        while(True):
            #Get a sample map from training library
            sample_map, _ = gen_lib.get()

            #Train GAN and generate a fake map
            generated_map = g.train(env, np.array(sample_map), classifier, opt, writer, idx)
            coded_fake_map = one_hot_to_ascii_level(generated_map.detach(), opt.token_list)
            ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

            #Skip if there is no reward in the map
            # if len(prize_map) == 0 or len(obs_x_list)>200:
            #     continue
            
            #Classifier makes prediction
            prediction = classifier.predict_label(torch.from_numpy((agent_map.reshape(1,3,opt.full_map_size,opt.full_map_size))).float())

            #Find best label for generated fake map
            rewards = []
            for i in range(3):
                reward = env.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                rewards.append(reward)
            actual = np.argmax(rewards)
            
            #Decide to add generated fake map into library
            if (prediction.item()==actual): #If classifier is correct - no need to add library
                continue
            else:
                #classifier is wrong - add it to library
                gen_lib.add(agent_map, actual, opt) 

                #Switch classifier into train mode and train with library(added with new map)
                classifier.train()

                #Retrained classifier with new updated library
                training_loss, trainc_labeled = classifier.trainer(gen_lib.train_library[0], gen_lib.train_library[1], optimizer)

                #Test classifier perf on test library
                classifier.eval()
                testc_labeled = classifier.predict(gen_lib.test_library)

                #log loss via tensorboard
                writer.add_scalar("trainc_labeled",trainc_labeled,idx+1)
                writer.add_scalar("testc_labeled",testc_labeled,idx+2)
                writer.add_scalar("training_loss", training_loss, idx+1)

                if len(gen_lib.train_library[0]) == opt.library_size:
                    g.better_save(len(gen_lib.train_library[0]))
                    break
                idx += 1
        #Close tensorboard logger    
        writer.close()
    
    elif(opt.mode == 'ganstyle_random'):        
        while(True):
            #Generate Random Map and add it to training library
            generated_map = generate_random_map(opt.full_map_size)
            ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate2(generated_map)

            ##Classifier makes prediction
            prediction = classifier.predict_label(torch.from_numpy((agent_map.reshape(1,3,opt.full_map_size,opt.full_map_size))).float())

            #Find best label for generated fake map
            rewards = []
            for i in range(3):
                reward = env.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                rewards.append(reward)
            actual = np.argmax(rewards)

            #Skip if there is no reward in the map
            # if len(prize_map) == 0 or len(obs_x_list)>200:
            #     continue
            
            #Decide to add generated fake map into library
            if (prediction.item()==actual): #If classifier is correct - no need to add library
                continue
            else:
                #classifier is wrong - add it to library
                gen_lib.add(agent_map, actual, opt) 

                #Switch classifier into train mode and train with library(added with new map)
                classifier.train()

                #Retrained classifier with new updated library
                training_loss, trainc_labeled = classifier.trainer(gen_lib.train_library[0], gen_lib.train_library[1], optimizer)

                #Load test library and log loss
                classifier.eval()
                testc_labeled = classifier.predict(gen_lib.test_library)

                #log loss via tensorboard
                writer.add_scalar("trainc_labeled",trainc_labeled,idx+1)
                writer.add_scalar("testc_labeled",testc_labeled,idx+2)
                writer.add_scalar("training_loss", training_loss, idx+1)

                if len(gen_lib.train_library[0]) == opt.library_size:
                    break
                idx += 1
        #Close tensorboard logger    
        writer.close()

    elif(opt.mode == 'random'):
        while(True):
            #Generate Random Map and add it to training library
            generated_map = generate_random_map(opt.full_map_size)
            ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list = fa_regenate2(generated_map)

            #Classifier makes prediction
            prediction = classifier.predict_label(torch.from_numpy((agent_map.reshape(1,3,opt.full_map_size,opt.full_map_size))).float())

            #Find best label for generated fake map
            rewards = []
            for i in range(3):
                reward = env.reset_and_step(ds_map, obstacle_map, prize_map, agent_map, map_lim, obs_y_list, obs_x_list, i+1)
                rewards.append(reward)
            actual = np.argmax(rewards)

            #Skip if there is no reward in the map
            # if len(prize_map) == 0 or len(obs_x_list)>200:
            #     continue

            #add random map to library
            gen_lib.add(agent_map, actual, opt)
            
            #Switch classifier into train mode and train with library(added with new map)
            classifier.train()

            #Retrained classifier with new updated library
            training_loss, trainc_labeled = classifier.trainer(gen_lib.train_library[0], gen_lib.train_library[1], optimizer)

            #Load test library and log loss
            classifier.eval()
            testc_labeled = classifier.predict(gen_lib.test_library)

            #log loss via tensorboard
            writer.add_scalar("trainc_labeled",trainc_labeled,idx+1)
            writer.add_scalar("testc_labeled",testc_labeled,idx+2)
            writer.add_scalar("training_loss", training_loss, idx+1)

            if len(gen_lib.train_library[0]) == opt.library_size:
                break
            idx += 1
        #Close tensorboard logger    
        writer.close()
    else:
        print("Unnoticeable Working Mode")

if __name__ == "__main__":
    main()
