#!/bin/bash

#######################
## Request resources ##
#######################
#SBATCH --account=def-lombaert
#SBATCH --nodes=1                 # number of nodes (computers)
#SBATCH --ntasks=1                # each line of execution in this file is one task
#SBATCH --gres=gpu:v100l:1        # 2 gpus per task
#SBATCH --cpus-per-task=16        # 16 cpus per task
#SBATCH --mem=64G                 # memory per node
#SBATCH --time=00:10:0            # DD-HH:MM:SS  
#SBATCH --output=%N-%j.out

source /home/arashash/projects/def-lombaert/arashash/Pawpularity/ENV/bin/activate
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html