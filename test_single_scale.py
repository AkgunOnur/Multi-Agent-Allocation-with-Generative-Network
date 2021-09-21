import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm

import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from environment.level_utils import group_to_token, one_hot_to_ascii_level, token_to_group
from environment.tokens import TOKEN_GROUPS as TOKEN_GROUPS
from models import calc_gradient_penalty, save_networks

from env_funcs import env_class
from classifier import LeNet
from read_maps import *
from construct_library import Library
import pandas as pd

stat_columns = ['errD_fake', 'errD_real', 'errG', 'reward']

def write_stats(stats,file_name='test_stats.csv'):
    df_stats = pd.DataFrame([stats], columns=stat_columns)
    df_stats.to_csv(file_name, mode='a', index=False,header=not os.path.isfile(file_name))

def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = torch.sqrt(F.mse_loss(real, z_prev))
    return opt.noise_update * RMSE


def test_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Test one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)
    real = reals[current_scale]

    if opt.game == 'environment':
        token_group = TOKEN_GROUPS

    nzx = real.shape[2]  # Noise size x
    nzy = real.shape[3]  # Noise size y

    padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

    if not opt.pad_with_noise:
        pad_noise = nn.ZeroPad2d(padsize)
        pad_image = nn.ZeroPad2d(padsize)
    else:
        pad_noise = nn.ReflectionPad2d(padsize)
        pad_image = nn.ReflectionPad2d(padsize)

    if current_scale == 0:  # Generate new noise
        z_opt = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        z_opt = pad_noise(z_opt)
    else:  # Add noise to previous output
        z_opt = torch.zeros([1, opt.nc_current, nzx, nzy]).to(opt.device)
        z_opt = pad_noise(z_opt)

    #logger.info("Training at scale {}", current_scale)\
    
    e = env_class(current_scale)
    lib = Library(opt.library_size)

    #Initalize classifier
    classifier = LeNet(numChannels=2, classes=8).to(opt.device) #(0-7) = 8 is max agent number in map
    #Load model and switch eval mode
    classifier.load_state_dict(torch.load("classifier.pt"))
    classifier.eval()



    for epoch in tqdm(range(opt.niter)):
        step = current_scale * opt.niter + epoch
        noise_ = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        noise_ = pad_noise(noise_)

        ############################
        # (1) D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # test with real
            D.zero_grad()
            output = D(real).to(opt.device)
            errD_real = -output.mean()

            # test with fake
            if (j == 0) & (epoch == 0):
                if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                    prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    input_from_prev_scale = prev
                    prev = pad_image(prev)
                    z_prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    z_prev = pad_noise(z_prev)
                    opt.noise_amp = 1
                else:  # First step in NOT the lowest scale
                    # We need to adapt our inputs from the previous scale and add noise to it
                    prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                       "rand", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        prev = group_to_token(prev, opt.token_list, token_group)

                    prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                    prev = pad_image(prev)
                    z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                         "rec", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        z_prev = group_to_token(z_prev, opt.token_list, token_group)

                    z_prev = interpolate(z_prev, real.shape[-2:], mode="bilinear", align_corners=False)
                    opt.noise_amp = update_noise_amplitude(z_prev, real, opt)
                    z_prev = pad_image(z_prev)
            else:  # Any other step
                prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)

                # For the seeding experiment, we need to transform from token_groups to the actual token
                if current_scale == (opt.token_insert + 1):
                    prev = group_to_token(prev, opt.token_list, token_group)

                prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                prev = pad_image(prev)

            # After creating our correct noise input, we feed it to the generator:
            noise = opt.noise_amp * noise_ + prev
            
            #Generate fake map
            G.zero_grad()
            fake = G(noise.detach(), prev.detach(), temperature=1 if current_scale != opt.token_insert else 1)

            # Then run the result through the discriminator
            output = D(fake.detach())
            errD_fake = output.mean()

            #Generate fake map(s) and make it playable
            coded_fake_map = one_hot_to_ascii_level(fake.detach(), opt.token_list)
            
            _, _, _, map, _, _, _ = fa_regenate(coded_fake_map)

            #Sent generated map into classifier and env
            #print("map shape: ", map)
            map = torch.from_numpy((map.reshape(1,2,80,80)))
            n_agents = classifier(map.float()) #classifier returns # agents that are deployed in map
            #print("n_agents: ", n_agents)
            #reset env and call D* for n_agents
            reward, zo_map = e.reset_and_step(coded_fake_map, n_agents+1)

            lib.evaluate(zo_map, reward)

            errG = -output.mean() + 0.05*torch.tensor(abs(reward), requires_grad=True)

            #======== log stats ===========
            write_stats([errD_fake, errD_real, errG, reward])
            #================================
    return z_opt, input_from_prev_scale, G
