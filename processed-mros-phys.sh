#!/bin/bash

# ut extract --file_regex './erda2/sleep-data/resources/mros/polysomnography/edfs/visit*/*.edf' --out_dir './erda2/sleep-data/resources/processed/mros/' --resample 128 --channels C3-M2 C4-M1 E1-M2 E2-M1 --log_dir logs-mros --continue_

# ut extract_hypno --file_regex './erda2/sleep-data/resources/mros/polysomnography/annotations-events-nsrr/visit*/*.xml' --out_dir './erda2/sleep-data/resources/processed/mros/' --log_dir logs-mros --continue_

ut cv_split --data_dir './erda2/sleep-data/resources/processed/mros/' --subject_dir_pattern 'mros*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --log_dir logs-mros --file_list

# ut extract --file_regex './erda2/sleep-data/resources/phys/tr*/*.mat' --out_dir './erda2/sleep-data/resources/processed/phys/' --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1 E1-M2 --log_dir logs-phys --continue_

# ut extract_hypno --file_regex './erda2/sleep-data/resources/phys/tr*/*.HYP.ids' --out_dir './erda2/sleep-data/resources/processed/phys/' --log_dir logs-phys --continue_

ut cv_split --data_dir './erda2/sleep-data/resources/processed/phys/' --subject_dir_pattern 'tr*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-phys --file_list