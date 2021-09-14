#!/bin/bash

#SBATCH -p longq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1

source activate toad

python /okyanus/users/deepdrone/Desktop/Multi-Agent-Allocation-with-Generative-Network/main_random.py
