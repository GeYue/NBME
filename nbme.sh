#!/bin/bash


python run_mlm.py \
--model_name_or_path "./model/huggingface-bert/microsoft/deberta-v3-large" \
--num_train_epochs 5 \
--train_file "train.json" \
--validation_file "val.json" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 8 \
--max_seq_length 384 \
--do_train \
--do_eval \
--output_dir "deberta-v3-large-regular384" \
--overwrite_output_dir \
--save_strategy "no" \
--logging_steps 200 \
--learning_rate 1e-5 \
--report_to "none"


<<COMMENT
python run_mlm.py \
--model_name_or_path "./model/huggingface-bert/microsoft/deberta-v2-xlarge" \
--num_train_epochs 5 \
--train_file "train.json" \
--validation_file "val.json" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--max_seq_length 384 \
--do_train \
--do_eval \
--output_dir "deberta-v2-xlarge-regular" \
--overwrite_output_dir \
--save_strategy "no" \
--logging_steps 200 \
--learning_rate 2e-6 \
--report_to "none"
COMMENT
