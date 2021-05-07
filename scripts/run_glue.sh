#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/

############# 76.1 ##########
python ../main.py \
    --task_name rte \
    --data_dir ./glue_data \
    --model_dir fine-tuned-tinybert/ \
    --model_save_dir output \
    --log_dir output

# python ../main.py \
#     --task_name cola \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name mnli \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name mrpc \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name qnli \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name sst-2 \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name sts-b \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name qqp \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output

# python ../main.py \
#     --task_name rte \
#     --data_dir ./glue_data \
#     --model_dir fine-tuned-tinybert/ \
#     --model_save_dir output \
#     --log_dir output
