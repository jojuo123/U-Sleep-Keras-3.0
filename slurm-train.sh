#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=U-Sleep-Keras3-train
#SBATCH --nodes=1
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64000M
#SBATCH -p gpu --gres=gpu:1
#number of cpus we want to allocate for each program
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-output/train/slurm-%j.out
#Skipping many options! see man sbatch
# From here on, we can start our program

# ./unmount_erda.sh
./mount_erda.sh
module load cuda/12.8
module load python/3.10.18
source /home/lht444/python-venv/usleep-keras/bin/activate
# pip install .
cd u-sleep-keras3
ut train --num_gpus 1 --max_loaded_per_dataset 40 --num_access_before_reload 32 --train_queue_type limitation --val_queue_type lazy --max_train_samples_per_epoch 1000000 --backend torch --overwrite
cd ..
# python -c "import keras; import os; os.environ['KERAS_BACKEND'] = 'torch'; print(keras.distribution.list_devices(device_type='gpu'))"
./unmount_erda.sh