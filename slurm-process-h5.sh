#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=U-Sleep-preprocess-h5
#SBATCH --nodes=1
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=20
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-output/process/slurm-%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program

./unmount_erda.sh
./mount_erda.sh
module load cuda/12.8
module load python/3.10.18
source /home/lht444/python-venv/usleep-keras/bin/activate
# pip install .
cd u-sleep-keras3
ut preprocess --out_path '/home/lht444/U-Sleep-Keras-3.0/erda2/sleep-data/resources/processed/processed_data.h5' --dataset_splits train_data val_data --log_file 'preprocessing_5' 
# python sanity_check.py
cd ..
./unmount_erda.sh
