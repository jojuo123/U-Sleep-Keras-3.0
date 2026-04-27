#!/bin/bash

# ut extract --file_regex './erda2/sleep-data/resources/dcsm/*/*.h5' --out_dir './erda2/sleep-data/resources/processed/dcsm/' --resample 128 --use_dir_names --channels 'F3-M2' 'F4-M1' 'C3-M2' 'C4-M1' 'O1-M2' 'O2-M1' 'E1-M2' 'E2-M2' --continue_ --log_dir logs-dcsm

ut extract_hypno --file_regex './erda2/sleep-data/resources/dcsm/*/*.ids' --out_dir './erda2/sleep-data/resources/processed/dcsm/' --log_dir logs-dcsm --overwrite

# ut cv_split --data_dir './erda2/sleep-data/resources/processed/dcsm/' --subject_dir_pattern 'tp*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --file_list --log_dir logs-dcsm

# ut extract --file_regex './erda2/sleep-data/resources/homepap/polysomnography/edfs/lab/*/*.edf' --out_dir './erda2/sleep-data/resources/processed/homepap/' --channels F4-M1 C4-M1 O2-M1 C3-M2 F3-M2 O1-M2 E1-M2 E2-M1 E1 E2  --resample 128 --log_dir logs-homepap --continue_

ut extract_hypno --file_regex './erda2/sleep-data/resources/homepap/polysomnography/annotations-events-nsrr/lab/*/*.xml' --out_dir './erda2/sleep-data/resources/processed/homepap/' --log_dir logs-homepap --overwrite --nsrr

# ut cv_split --data_dir './erda2/sleep-data/resources/processed/homepap/' --subject_dir_pattern 'homepap*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*-(\d+)' --file_list --log_dir logs-homepap