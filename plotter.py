from plot_utils import *
from train import curriculum_design
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plotter for the Train Files')

    parser.add_argument('--file_loc',type=str, default="output_nocur_10", help='location of file to make the plot')
   
    args = parser.parse_args()

    nocur_folder = "seko_models/" + args.file_loc + "/output_nocur_10"
    level_list = ["easy", "medium", "target"]    
    map_lim = 10
    for index in range(5):
        fig = plt.figure(figsize=(15, 15))
        for i, level in enumerate(level_list):
            title = "Map " + str(index) + " - " + level
            file_folder = "maps/rendered_" + str(map_lim) + "x" + str(map_lim) + "/" + level + "/map_" + str(index) + ".png"
        plot_nocurr(folder = nocur_folder,  index=index)

  