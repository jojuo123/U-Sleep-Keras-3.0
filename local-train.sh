#!/bin/bash

./unmount_erda.sh
./mount_erda.sh
source /python-venv/usleep-keras3/bin/activate
pip install .
cd u-sleep-keras3
# ut train --num_gpus 1 --preprocessed --max_train_samples_per_epoch 1000000 --backend torch --continue_training --no_warnings
ut train --num_gpus 1 --max_loaded_per_dataset 50 --num_access_before_reload 32 --train_queue_type limitation --val_queue_type lazy --max_train_samples_per_epoch 1000000 --n_processes 20 --backend torch --continue_training --no_warnings
cd ..

./unmount_erda.sh