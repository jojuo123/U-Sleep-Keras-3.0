#!/bin/bash

# ut extract --file_regex './erda2/sleep-data/resources/cfs/polysomnography/edfs/*.edf' --out_dir './erda2/sleep-data/resources/processed/cfs/' --resample 128 --channels EEG1 EEG2 EEG3 EOG-L EOG-R --channels C3-A2 C4-A1 LOC-A2 ROC-A1 --overwrite --log_dir logs-cfs

# ut extract_hypno --file_regex './erda2/sleep-data/resources/cfs/polysomnography/annotations-events-nsrr/*.xml' --out_dir './erda2/sleep-data/resources/processed/cfs/' --log_dir logs-cfs --overwrite --nsrr

ut cv_split --data_dir './erda2/sleep-data/resources/processed/cfs/' --subject_dir_pattern 'cfs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'cfs-visit.*' --log_dir logs-cfs --file_list
