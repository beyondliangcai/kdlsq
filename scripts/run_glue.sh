export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
python lsq/main.py --task_name="rte" --data_dir="./glue-data"  --model_dir="finetuned-bert/" --model_save_dir="output_model"  --log_dir="output"
#run kdlsq glue
python kdlsq_glue/run_kdlsq_glue.py  --model_dir="finetuned-bert/" --data_dir="./glue-data"--data_dir="./glue-data"
