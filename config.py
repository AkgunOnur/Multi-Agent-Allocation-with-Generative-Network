# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
import numpy as np
import torch


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)




def get_arguments():
    parser = argparse.ArgumentParser()
    # Game Type - Supports: environment
    parser.add_argument("--game", default="environment", help="Which game is to be used?")

    #Working mode - Supports: train and test
    parser.add_argument("--mode", default="train", help="working train mode selection")
    parser.add_argument("--testmode", default="test_gan", help="working test mode selection")
    parser.add_argument("--test_type", default="gan", help="Testing with test library maps or gan generated maps")
    parser.add_argument("--library_size", type=int, default=25, help=" final training library size")

    # workspace:
    parser.add_argument("--not_cuda", action="store_true", help="disables cuda", default=0)

    # load, input, output, save configurations:
    parser.add_argument("--netG", default="", help="path to netG (to continue training)")
    parser.add_argument("--netD", default="", help="path to netD (to continue training)")
    parser.add_argument("--manualSeed", type=int, help="manual seed")
    parser.add_argument("--out", help="output folder", default="output")
    parser.add_argument("--input-dir", help="input image dir", default="input")
    parser.add_argument("--input-name", help="input image name", default="harita.txt")
    parser.add_argument("--test_dir", help="test input map dir", default="test_maps")

    # networks hyper parameters:
    parser.add_argument("--nfc", type=int, help="number of filters for conv layers", default=64)
    parser.add_argument("--ker_size", type=int, help="kernel size", default=3)
    parser.add_argument("--num_layer", type=int, help="number of layers", default=3)

    # scaling parameters:
    parser.add_argument("--scales", nargs='+', type=float, help="Scales descending (< 1 and > 0)",
                        default=[1])#default=[0.75, 0.5, 0.25]
    parser.add_argument("--full_map_size", type=int, default=40, help="Full map size. Default 80x80")
    parser.add_argument("--noise_update", type=float, help="additive noise weight", default=0.1)
    parser.add_argument("--pad_with_noise", type=bool, help="use reflection padding? (makes edges random)",
                        default=False)

    # optimization hyper parameters:
    parser.add_argument("--niter", type=int, default=15, help="number of epochs to train per scale")
    parser.add_argument("--gamma", type=float, help="scheduler gamma", default=0.1)
    parser.add_argument("--lr_g", type=float, default=0.0005, help="learning rate, default=0.0005")
    parser.add_argument("--lr_d", type=float, default=0.0005, help="learning rate, default=0.0005")
    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5")
    parser.add_argument("--Gsteps", type=int, help="Generator inner steps", default=1)
    parser.add_argument("--Dsteps", type=int, help="Discriminator inner steps", default=1)
    parser.add_argument("--lambda_grad", type=float, help="gradient penalty weight", default=0.1)
    # alpha controls how much the reconstruction factors into the training. 0 = No reconstruction.
    parser.add_argument("--alpha", type=float, help="reconstruction loss weight", default=100)

    # possible token grouping (Experimental Feature! May break Everything!)
    parser.add_argument("--token_insert", type=int, help="layer in which token groupings will be split out "
                                                         "(<-2 means no grouping at all)", default=-2)

    return parser


def post_config(opt):
    """ Initializes parameters. We're using Namespace opt to pass a lot of used parameters to many functions. """

    opt.device = torch.device("cpu")# if opt.not_cuda else "cuda:0")
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    set_seed(opt.manualSeed)

    # Defaults for other namespace values that will be overwritten during runtime
    opt.nc_current = 3  # n tokens of level 1-1
    if not hasattr(opt, "out_"):
        opt.out_ = "%s/%s/" % (opt.out, opt.input_name[:-4])
    opt.outf = "0"  # changes with each scale trained
    opt.inf = "0"  # changes with each scale tested
    opt.num_scales = len(opt.scales)  # number of scales is implicitly defined
    opt.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
    opt.seed_road = None  # for mario kart seed roads after training
    opt.token_list = ['-', 'W', 'X']  # default list of easy1.txt
    opt.ImgGen = []  # needs to be set to the correct image gen for each game
    opt.stop_scale = opt.num_scales  # which scale to stop on - usually always last scale defined

    return opt
