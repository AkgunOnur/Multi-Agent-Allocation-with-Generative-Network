#!/bin/bash

#SBATCH -p defq
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 1

source activate swarm

python /okyanus/users/deepdrone/Multi-Agent-Allocation-with-Generative-Network/main.py --test
