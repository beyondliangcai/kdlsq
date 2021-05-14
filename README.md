# kdlsq
KDLSQ-BERT model combines knowledge distillation (KD) with learned step size quantization (LSQ) for language model quantization. The main idea of our method is that the KD technique is leveraged to transfer the knowledge from a ”teacher” model to a ”student” model when exploiting LSQ to quantize that ”student” model during the quantization training process
# Train LSQ-BERT without kd on GLUE and SQUAD tasks
- train LSQ-BERT on GLUE tasks:
```text

python lsq/main.py \
        --task_name=${TASK_NAME}\
        --data_dir=${GLUE_DIR} \
        --model_dir=${MODEL_DIR} \
        --model_save_dir=${MODEL_SAVE_DIR} \
        --log_dir=${LOG_DIR}

```        

- train LSQ-BERT on SQUAD tasks, we put the squad model and data into the same place which was set in the code question_answering.py:
```text

python lsq/question_answering.py \
        --model_save_dir=${MODEL_SAVE_DIR} \
        --log_dir=${LOG_DIR}

```

# Train KDLSQ-BERT with kd on GLUE and SQUAD tasks
we put the model and data  into the same place which was set in the code run_kdlsq_glue.py and run_kdlsq_qa.py:
- train KDLSQ-BERT on GLUE tasks:
```text

python kdlsq_glue/run_kdlsq_glue.py

```
- train KDLSQ-BERT on SQUAD tasks:
```text

python kdlsq_qa/run_kdlsq_qa.py

```
# To Dos
- optimized code, we will implemented KDLSQ-BERT with Huawei MindSpore AI framework.
- Support Horovod for efficient distributed learning.

# Reference
Jing Jin, Cai Liang, Tiancheng Wu, Liqin Zou, Zhiliang Gan
[KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with Learned Step Size Quantization
](https://arxiv.org/abs/2101.05938)
```
@article{jin2021kdlsq,
  title={Kdlsq-bert: A quantized bert combining knowledge distillation with learned step size quantization},
  author={Jin, Jing and Liang, Cai and Wu, Tiancheng and Zou, Liqin and Gan, Zhiliang},
  journal={arXiv preprint arXiv:2101.05938},
  year={2021}
}
```