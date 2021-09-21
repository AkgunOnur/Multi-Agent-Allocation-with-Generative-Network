# Code inspired by https://github.com/tamarott/SinGAN
import os

import torch
import wandb
from tqdm import tqdm

from models import init_models, reset_grads, restore_weights
from environment.level_utils import load_level_from_text
#from dqn_model import *
from point_mass_formation import AgentFormation
from env_funcs import env_class
import os
import glob

from environment.level_utils import one_hot_to_ascii_level, token_to_group
from environment.tokens import TOKEN_GROUPS as TOKEN_GROUPS
from environment.special_downsampling import special_downsampling
from models import init_models, reset_grads, restore_weights
from models.generator import Level_GeneratorConcatSkip2CleanAdd
from test_single_scale import test_single_scale

from environment.level_image_gen import LevelImageGen as LevelGen
from environment.special_downsampling import special_downsampling
from environment.level_utils import read_level, read_level_from_file
from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS


def test(opt):
    """ Wrapper function for testubg. Get test maps and then calls test_single_scale on each. """

    opt.scales.insert(0, 1)

    test_map_dir = os.listdir(opt.test_dir)
    test_map_dir.sort()

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'environment':
        opt.ImgGen = LevelGen(sprite_path)
        replace_tokens = REPLACE_TOKENS
        #downsample = special_downsampling
    else:
        NameError("name of --game not recognized. Supported: environment")

    #Test with test_map library
    if(opt.test_type == 'library'):
        # Test Loop
        for directory in test_map_dir:
            current_dir = os.path.join(opt.test_dir, directory +"/")
            #print("current_dir: ", current_dir)
            file_names = glob.glob("./"+current_dir+"*.txt")

            scale_number = 0#int(int(directory)/20-1)
            #print("scale_number: ", scale_number)

            #initalizerl agent and load its weights
            e = env_class(int(scale_number))

            agent_mean_reward = 0.0

            #for maps in this scale
            for i in range(len(file_names)):
                map = load_level_from_text(file_names[i])
                
                #Deploy agent in map and get reward for couple of iterations
                agent_mean_reward += e.test(map)
                #print("Map: "+ str(file_names[i])+ " agent_mean_reward: ", agent_mean_reward)
                #log rewards
            print("agent_mean_reward for scale" + str(scale_number) + " is :", agent_mean_reward/len(file_names))
        
    #Test with GAN generated maps
    elif(opt.test_type == 'gan'):
        generators = []
        noise_maps = []
        noise_amplitudes = []

        token_group = TOKEN_GROUPS

        scales = [[x, x] for x in opt.scales]
        opt.num_scales = len(scales)

        real = read_level(opt, None, replace_tokens).to(opt.device)
        scaled_list = special_downsampling(opt.num_scales, scales, real, opt.token_list)
        reals = [*scaled_list, real]

        if opt.token_insert >= 0:
            reals = [(token_to_group(r, opt.token_list, token_group) if i < opt.token_insert else r) for i, r in enumerate(reals)]
            reals.insert(opt.token_insert, token_to_group(reals[opt.token_insert], opt.token_list, token_group))
            input_from_prev_scale = torch.zeros_like(reals[0])

        stop_scale = len(reals)
        opt.stop_scale = stop_scale

        #for each scale
        for current_scale in range(0, stop_scale):
            #Initialize GAN and Discriminator and frezee them
            opt.inf = "%s/%d" % (opt.out_, current_scale)
            if current_scale < (opt.token_insert + 1):  # (stop_scale - 1):
                opt.nc_current = len(token_group)

            # Initialize models
            D, G = init_models(opt)
            
            #Experimental
            D, G = restore_weights(D, G, current_scale, opt)

            #Reset grads and switch to evaluation
            G = reset_grads(G, False)
            G.eval()
            D = reset_grads(D, False)
            D.eval()

            z_opt, input_from_prev_scale, G = test_single_scale(D,  G, reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt)
            #Generate fake maps with gan
            #test rl agent on this map
            #log reward of agent and loss of generator and discriminator

            generators.append(G)
            noise_maps.append(z_opt)
            noise_amplitudes.append(opt.noise_amp)

            del D, G
    else:
        print("Unavailable test mode!! Options:['library', 'gan']")
################################################################################################