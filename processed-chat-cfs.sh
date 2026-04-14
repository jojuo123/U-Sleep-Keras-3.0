#!/bin/bash

ut extract --log_dir logs-chat --overwrite --file_regex './erda2/sleep-data/resources/chat/polysomnography/edfs/*/*.edf' --out_dir './erda2/sleep-data/resources/processed/chat/' --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 T3-M2 T4-M1 O1-M2 O2-M1 E1-M2 E2-M1

ut extract_hypno --log_dir logs-chat --overwrite --file_regex './erda2/sleep-data/resources/chat/polysomnography/annotations-events-nsrr/*/*.xml' --out_dir './erda2/sleep-data/resources/processed/chat/' 

ut cv_split --data_dir './erda2/sleep-data/resources/processed/chat/' --subject_dir_pattern 'chat*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)'
