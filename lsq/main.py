import os
import logging
import argparse
import random
import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sequence_classification import TransformerSequenceClassifier
from lsq.glue_tasks import (convert_examples_to_features, max_seq_lengths, get_glue_task, output_modes,
                            get_metric_fn, batch_sizes, get_metric_key, eval_steps)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#from transformers import set_seed

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids)
    return tensor_data, all_label_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default='cola',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default='data/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models/',
                        type=str,
                        help="The model dir.")

    parser.add_argument("--root_dir", default='./', type=str)
    parser.add_argument("--log_dir", default='', type=str)
    parser.add_argument("--tensorboard_dir", default='', type=str)
    parser.add_argument("--model_save_dir", default='./', type=str)
    
    args = parser.parse_args()
    set_seed(42)
    logging.info('The args: {}'.format(args))
    task_name = args.task_name

    data_dir = os.path.join(args.root_dir,args.data_dir,task_name)
    model_dir = os.path.join(args.root_dir,args.model_dir,task_name)

    task = get_glue_task(task_name,data_dir)
    labels = task.get_labels()
    output_mode = output_modes[task_name]
    max_seq_length = max_seq_lengths[task_name]
    batch_size = batch_sizes[task_name]
    eval_step = eval_steps[task_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    model = TransformerSequenceClassifier(model_type='quant_bert', 
                            labels=labels,task_type=output_mode,metric_fn=get_metric_fn(task_name),metric_key=get_metric_key(task_name),
                            model_name_or_path=model_dir,
                            load_quantized=False,config_name=model_dir, tokenizer_name=model_dir,
                            do_lower_case=True,output_path=args.model_save_dir,
                            device=device,n_gpus=n_gpus)

    tokenizer = model.tokenizer

    train_examples = task.get_train_examples()
    train_features = convert_examples_to_features(train_examples,labels,max_seq_length=max_seq_length,
                                                tokenizer=tokenizer,output_mode=output_mode)
    train_data, _ = get_tensor_data(output_mode,train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    dev_examples = task.get_dev_examples()
    dev_features = convert_examples_to_features(dev_examples,labels,max_seq_length=max_seq_length,
                                                tokenizer=tokenizer,output_mode=output_mode)
    dev_data, _ = get_tensor_data(output_mode,dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    model.train(train_data_set=train_dataloader,dev_data_set=dev_dataloader,logging_steps=eval_step)

    print()

if __name__ == "__main__":
    main()
