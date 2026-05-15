#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=U-Sleep-Keras3-train
#SBATCH --nodes=1
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#SBATCH -p gpu --gres=gpu:titanrtx:1
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-output/sanity/slurm-%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program

# ./unmount_erda.sh
# ./mount_erda.sh
module load cuda/12.6
module load python/3.12.8
source /home/lht444/python-venv/new-usleep/bin/activate
python sanity_check_keras.py

# ./unmount_erda.sh