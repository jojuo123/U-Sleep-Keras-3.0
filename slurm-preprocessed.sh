#!/bin/bash
#The partition is the queue you want to run on. standard is gpu and can be ommitted.
#SBATCH --job-name=U-Sleep-process
#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1
#number of cpus we want to allocate for each program
#SBATCH --cpus-per-task=64
#We expect that our program should not run longer than 2 days
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=2-00:00:00
#Skipping many options! see man sbatch
# From here on, we can start our program

./mount_erda.sh
module load cuda/12.8
module load python/3.10.18
source /home/lht444/python-venv/usleep-keras/bin/activate
./processed-chat-cfs.sh
./unmount_erda.sh
