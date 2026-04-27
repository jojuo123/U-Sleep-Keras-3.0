#!/bin/bash

ut extract --file_regex './erda2/sleep-data/resources/sedf_sc/SC*/*PSG.edf' --out_dir './erda2/sleep-data/resources/processed/sedf_sc/' --resample 128 --channels 'EEG Fpz-Cz' 'EEG Pz-Oz' 'EOG horizontal' --rename Fpz-Cz Pz-Oz EOG --log_dir logs-sedf-sc --overwrite --use_dir_names

ut extract_hypno --file_regex './erda2/sleep-data/resources/sedf_sc/SC*/*Hypnogram.edf' --out_dir './erda2/sleep-data/resources/processed/sedf_sc/' --log_dir logs-sedf-sc --overwrite

ut cv_split --data_dir './erda2/sleep-data/resources/processed/sedf_sc/' --subject_dir_pattern 'SC*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'SC4(\d{2}).*' --log_dir logs-sedf-sc --file_list

ut extract --file_regex './erda2/sleep-data/resources/sedf_st/ST*/*PSG.edf' --out_dir './erda2/sleep-data/resources/processed/sedf_st/' --resample 128 --channels 'EEG Fpz-Cz' 'EEG Pz-Oz' 'EOG horizontal' --rename Fpz-Cz Pz-Oz EOG --log_dir logs-sedf-st --overwrite --use_dir_names

ut extract_hypno --file_regex './erda2/sleep-data/resources/sedf_st/ST*/*Hypnogram.edf' --out_dir './erda2/sleep-data/resources/processed/sedf_st/' --fill_blanks 'Sleep stage ?' --log_dir logs-sedf-st --overwrite

ut cv_split --data_dir './erda2/sleep-data/resources/processed/sedf_st/' --subject_dir_pattern 'ST*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'ST7(\d{2}).*' --log_dir logs-sedf-st --file_list