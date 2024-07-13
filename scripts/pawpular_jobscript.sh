#!/bin/bash

#######################
## Request resources ##
#######################
#SBATCH --account=def-chdesa
#SBATCH --requeue   
#SBATCH --array=1-10              # submit n independant jobs in parrallel using arrayjobs mechanism
#SBATCH --nodes=1                 # number of nodes (computers)
#SBATCH --ntasks=1                # each line of execution in this file is considered as one task
#SBATCH --gres=gpu:4              # gpus per task
#SBATCH --cpus-per-task=16        # cpus per task
#SBATCH --mem=128G                # memory per node
#SBATCH --time=08:00:0            # DD-HH:MM:SS  
#SBATCH --output=/home/arashash/projects/def-lombaert/arashash/Pawpularity/tmp_outputs/%N-%A_%a.out

###########
## Setup ##
###########
source /home/arashash/projects/def-lombaert/arashash/Pawpularity/ENV_new_timm/bin/activate # activate the virtual env

# set the environment variables
export NCCL_BLOCKING_WAIT=1 # set this variable to avoid timeout errors in Pytorch Lightning
export CUDA_VISIBLE_DEVICES=0,1,2,3 # assign ids to gpus
export WANDB_API_KEY="" # wandb login info
export WANDB_USERNAME=""
export WANDB_CACHE_DIR="/scratch/arashash/Pawpularity/tmp/"
export WANDB_MODE=offline

# copy and unzip data from the login node to the compute node
cd $SLURM_TMPDIR
tar -xf /home/arashash/projects/def-lombaert/arashash/Pawpularity/datasets/datasets.tar --strip-components=8

##############
## Training ##
##############
cd /home/arashash/projects/def-lombaert/arashash/Pawpularity/repo/src/
python pawpular_training.py --data_dir "${SLURM_TMPDIR}/" --repo_dir "/home/arashash/projects/def-lombaert/arashash/Pawpularity/repo/" --save_dir "/scratch/arashash/Pawpularity/tmp/" --slurm_array_id $SLURM_ARRAY_TASK_ID --devices 0 1 2 3


