#!/bin/bash

ut extract --file_regex './erda2/sleep-data/resources/shhs/polysomnography/edfs/shhs*/*.edf' --out_dir './erda2/sleep-data/resources/processed/shhs/' --channels 'EEG' 'EEG(sec)' 'EOG(L)' 'EOG(R)' --resample 128 --rename_channels 'C4-A1' 'C3-A2' 'EOG(L)-PG1' 'EOG(R)-PG1' --log_dir logs-shhs --overwrite

ut extract_hypno --file_regex './erda2/sleep-data/resources/shhs/polysomnography/annotations-events-nsrr/shhs*/*.xml' --out_dir './erda2/sleep-data/resources/processed/shhs/' --log_dir logs-shhs --overwrite

ut cv_split --data_dir './erda2/sleep-data/resources/processed/shhs/' --subject_dir_pattern 'shhs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-(.*)' --log_dir logs-shhs

ut extract --file_regex './erda2/sleep-data/resources/sof/polysomnography/edfs/shhs*/*.edf' --out_dir './erda2/sleep-data/resources/processed/shhs/' --channels C3-A2 C4-A1 LOC-A2 ROC-A1 --resample 128 --log_dir logs-sof --overwrite

ut extract_hypno --file_regex './erda2/sleep-data/resources/sof/polysomnography/annotations-events-nsrr/*.xml' --out_dir './erda2/sleep-data/resources/processed/sof/' --log_dir logs-sof --overwrite

ut cv_split --data_dir './erda2/sleep-data/resources/processed/sof/' --subject_dir_pattern 'sof*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-sof