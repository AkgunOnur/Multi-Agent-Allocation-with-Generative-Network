import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm


from generate_noise import generate_spatial_noise
from environment.level_utils import  one_hot_to_ascii_level
from models import init_models, reset_grads, calc_gradient_penalty, save_networks
from draw_concat import draw_concat

from read_maps import *
import pandas as pd


stat_columns = ['errD_fake', 'errD_real', 'errG']

def write_stats(stats,file_name='errors.csv'):
    df_stats = pd.DataFrame([stats], columns=stat_columns)
    df_stats.to_csv(file_name, mode='a', index=False,header=not os.path.isfile(file_name))

class GAN:
    def __init__(self,opt):
        self.D, self.G = init_models(opt)

        self.padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

        self.pad_noise = nn.ZeroPad2d(self.padsize)
        self.pad_image = nn.ZeroPad2d(self.padsize)

        # setup optimizer
        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.D.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        self.schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerD, milestones=[1500, 2500], gamma=opt.gamma)
        self.schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizerG, milestones=[1500, 2500], gamma=opt.gamma)

    def train(self, e, real, classifier, opt):
        """ Train one scale. D and G are the discriminator and generator, real is the original map and its label.
        opt is a namespace that holds all necessary parameters. """
        real = torch.FloatTensor(real)
        nzx = real.shape[2]  # Noise size x
        nzy = real.shape[3]  # Noise size y

        for step in tqdm(range(opt.niter)):
            noise_ = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
            noise_ = self.pad_noise(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                #==========================================
                if(j==0 and step==0):
                    prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    prev = self.pad_image(prev)
                else:
                    prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                    prev = self.pad_image(prev)
                #==========================================
                # train with real nad fake
                self.D.zero_grad()
                output_r = self.D(real).to(opt.device)
                errD_real = -output_r.mean()
                
                errD_real.backward(retain_graph=True)

                # After creating our correct noise input, we feed it to the generator:
                noise = opt.noise_amp * noise_ + prev
                fake = self.G(noise.detach(), prev, temperature=1)

                # Then run the result through the discriminator
                output_f = self.D(fake.detach())
                errD_fake = output_f.mean()

                # Backpropagation
                errD_fake.backward(retain_graph=False)
                # Gradient Penalty
                gradient_penalty = calc_gradient_penalty(self.D, real, fake, opt.lambda_grad, opt.device)
                gradient_penalty.backward(retain_graph=False)

                self.optimizerD.step()

            ###########################
            # (2) Update G network: maximize D(G(z))
            ###########################
            for j in range(opt.Gsteps):
                self.G.zero_grad()
                fake = self.G(noise.detach(), prev.detach(), temperature=1)
                output = self.D(fake)
                #================================
                #Generate fake map(s) and make it playable
                coded_fake_map = one_hot_to_ascii_level(fake.detach(), opt.token_list)
                # print("coded_fake:", coded_fake_map)
                ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list = fa_regenate(coded_fake_map)

                #Sent generated map into classifier and env
                prediction = classifier.predict2(torch.from_numpy((harita.reshape(1,3,40,40))).float())#+1
                #reset env and call D* for n_agents
                rewards = []
                for i in range(6):
                    reward = e.reset_and_step(ds_map, obstacle_map, prize_map, harita, map_lim, obs_y_list, obs_x_list, i+1)
                    rewards.append(reward)
                #Get actual best n_agents
                actual = np.argmax(rewards)#+1

                #Compute generator error
                errG = -output.mean() + torch.tensor(abs(prediction-actual))
                #print("errG: ", errG)
                errG.backward(retain_graph=False)
                self.optimizerG.step()
            
            #======== log stats ===========
            write_stats([errD_fake.item(), errD_real.item(), errG.item()])
            #================================

            # Learning Rate scheduler step
            self.schedulerD.step()
            self.schedulerG.step()

        #Take G,D
        self.G = reset_grads(self.G, True)
        self.D = reset_grads(self.D, True)
        with torch.no_grad():
            generated_map = self.G(noise.detach(), prev.detach(), temperature=1)
        return generated_map
    
    def better_save(self, iteration):
        save_networks(self.G, self.D, iteration)
