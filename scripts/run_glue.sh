export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
python ../main.py --task_name="rte" --data_dir="./glue_data"  --model_dir="fine-tuned-bert/" --model_save_dir="output_model"  --log_dir="output"
