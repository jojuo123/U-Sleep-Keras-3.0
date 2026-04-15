#!/bin/bash

# ut extract --file_regex './erda2/sleep-data/resources/abc/polysomnography/edfs/*/*.edf' --out_dir './erda2/sleep-data/resources/processed/abc/' --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1 E1-M2 E2-M1 --log_dir logs-abc --continue_

# ut extract_hypno --file_regex './erda2/sleep-data/resources/abc/polysomnography/annotations-events-nsrr/*/*.xml' --out_dir './erda2/sleep-data/resources/processed/abc/' --log_dir logs-abc

# ut cv_split --data_dir './erda2/sleep-data/resources/processed/abc/' --subject_dir_pattern 'abc*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --log_dir logs-abc --file_list

ut extract --file_regex './erda2/sleep-data/resources/ccshs/polysomnography/edfs/*.edf' --out_dir './erda2/sleep-data/resources/processed/ccshs/' --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1 --log_dir logs-ccshs --continue_

ut extract_hypno --file_regex './erda2/sleep-data/resources/ccshs/polysomnography/annotations-events-nsrr/*.xml' --out_dir './erda2/sleep-data/resources/processed/ccshs/' --log_dir logs-ccshs

ut cv_split --data_dir './erda2/sleep-data/resources/processed/ccshs/' --subject_dir_pattern 'ccshs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-ccshs --file_list