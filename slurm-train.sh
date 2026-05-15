#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=U-Sleep-Keras3-train
#SBATCH --nodes=1
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=40000M
#SBATCH -p gpu --gres=gpu:a100:1
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-output/train/slurm-%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program

./unmount_erda.sh
./mount_erda.sh
module load cuda/12.8
module load python/3.10.18
source /home/lht444/python-venv/usleep-keras/bin/activate
pip install .
cd u-sleep-keras3
# export PYTORCH_ALLOC_CONF=gc_threshold:0.6,segment_size_mb:16
ut train --num_gpus 1 --preprocessed --max_train_samples_per_epoch 2000000 --backend torch --continue_training
cd ..

# cd test_scheduler
# # export PYTORCH_ALLOC_CONF=gc_threshold:0.6,segment_size_mb:16
# ut train --num_gpus 1 --preprocessed --max_train_samples_per_epoch 1000 --backend torch --overwrite
# cd ..

./unmount_erda.sh