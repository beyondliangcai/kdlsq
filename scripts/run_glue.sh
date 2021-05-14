export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/
#run lsq
python lsq/main.py --task_name="rte" --data_dir="./glue-data"  --model_dir="finetuned-bert/" --model_save_dir="output_model"  --log_dir="output"
python lsq/question_answering.py --model_save_dir="output_model"  --log_dir="output"
#run kdlsq
python kdlsq_glue/run_kdlsq_glue.py
python kdlsq_qa/run_kdlsq_qa.py

