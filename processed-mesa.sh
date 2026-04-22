#!/bin/bash

# ut extract --file_regex './erda2/sleep-data/resources/mesa/polysomnography/edfs/*.edf' --out_dir './erda2/sleep-data/resources/processed/mesa/' --resample 128 --channels EEG1 EEG2 EEG3 EOG-L EOG-R --rename Fz-Cz Cz-Oz C4-M1 E1-FPz E2-FPz --log_dir logs-mesa --continue_

# ut extract_hypno --file_regex './erda2/sleep-data/resources/mesa/polysomnography/annotations-events-nsrr/*.xml' --out_dir './erda2/sleep-data/resources/processed/mesa/' --log_dir logs-mesa --overwrite

ut cv_split --data_dir './erda2/sleep-data/resources/processed/mesa/' --subject_dir_pattern 'mesa*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-mesa --file_list
