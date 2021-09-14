# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
from train import train
#from test import test

from environment.tokens import REPLACE_TOKENS as REPLACE_TOKENS

from environment.level_image_gen import LevelImageGen as LevelGen
from environment.special_downsampling import special_downsampling
from environment.level_utils import read_level, read_level_from_file

from config import get_arguments, post_config
from loguru import logger
import wandb
import sys

def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    return [opt.input_name.split(".")[0]]


def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """

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
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'environment':
        opt.ImgGen = LevelGen(sprite_path)
        replace_tokens = REPLACE_TOKENS
        #downsample = special_downsampling
    else:
        NameError("name of --game not recognized. Supported: environment")


    if(opt.mode == 'train'):
        #TODO: Buraya for ekleyip birden fazla resimle egitim yapilabilir.
        
        # Read level according to input arguments
        real = read_level(opt, None, replace_tokens).to(opt.device)

        # Train!
        generators, noise_maps, reals, noise_amplitudes = train(real, opt)

        
        # Generate Samples of same size as level
        logger.info("Finished training! Generating random samples...")
        in_s = None
        generate_samples(generators, noise_maps, reals,
                        noise_amplitudes, opt, in_s=in_s)
    
    elif(opt.mode == 'test'):
        test(opt)
    else:
        print("Unnoticeable Working Mode")

if __name__ == "__main__":
    main()
