#!/bin/bash

ut cv_split --data_dir './erda2/sleep-data/resources/processed/abc/' --subject_dir_pattern 'abc*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --log_dir logs-abc --file_list --overwrite

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/ccshs/' --subject_dir_pattern 'ccshs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-ccshs --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/cfs/' --subject_dir_pattern 'cfs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'cfs-visit.*' --log_dir logs-cfs --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/chat/' --subject_dir_pattern 'chat*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/dcsm/' --subject_dir_pattern 'tp*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --file_list --log_dir logs-dcsm

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/homepap/' --subject_dir_pattern 'homepap*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*-(\d+)' --file_list --log_dir logs-homepap

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/mesa/' --subject_dir_pattern 'mesa*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-mesa --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/mros/' --subject_dir_pattern 'mros*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)' --log_dir logs-mros --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/phys/' --subject_dir_pattern 'tr*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-phys --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/sof/' --subject_dir_pattern 'sof*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --log_dir logs-sof --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/shhs/' --subject_dir_pattern 'shhs*-*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-(.*)' --log_dir logs-shhs --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/sedf_st/' --subject_dir_pattern 'ST*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'ST7(\d{2}).*' --log_dir logs-sedf-st --file_list

ut cv_split --overwrite --data_dir './erda2/sleep-data/resources/processed/sedf_sc/' --subject_dir_pattern 'SC*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'SC4(\d{2}).*' --log_dir logs-sedf-sc --file_list