#!/bin/bash

ut extract --file_regex '/home/lht444/erda/sleep-data/resources/chat/polysomnography/edfs/*/*.edf' --out_dir '/home/lht444/erda/sleep-data/resources/processed/chat/' --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 T3-M2 T4-M1 O1-M2 O2-M1 E1-M2 E2-M1 --log_dir logs-chat

ut extract_hypno --file_regex '/home/lht444/erda/sleep-data/resources/chat/polysomnography/annotations-events-nsrr/*/*.xml' --out_dir '${local_path}/processed/chat/' --log_dir logs-chat

ut cv_split --data_dir '/home/lht444/erda/sleep-data/resources/processed/chat/' --subject_dir_pattern 'chat*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --log_dir logs-chat

ut extract --file_regex '/home/lht444/erda/sleep-data/resources/cfs/polysomnography/edfs/*.edf' --out_dir '/home/lht444/erda/sleep-data/resources/processed/cfs/' --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1 --overwrite --log_dir

ut extract_hypno --file_regex '/home/lht444/erda/sleep-data/resources/cfs/polysomnography/annotations-events-nsrr/*.xml' --out_dir '/home/lht444/erda/sleep-data/resources/processed/cfs/'

ut cv_split --data_dir '/home/lht444/erda/sleep-data/resources/processed/cfs/' --subject_dir_pattern 'cfs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*famID(.*)'
