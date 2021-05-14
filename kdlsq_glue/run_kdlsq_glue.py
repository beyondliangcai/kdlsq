from __future__ import absolute_import, division, print_function

import time
import argparse
import csv
import logging
import os
import random
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformer.modeling import BertConfig, BertForSequenceClassification  # it's for teacher model
from quantized_bert import QuantizedBertConfig, QuantizedBertForSequenceClassification  # it's for student model
from quantization import reset_wgt_alpha, kl_divergence_initilization
from quantization import reset_act_alpha_true, reset_act_alpha_false

from transformer.tokenization import BertTokenizer
from Q8BertOptimizer.optimization import AdamW, get_linear_schedule_with_warmup
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME

MODEL_CONFIGURATIONS = {
    "bert": (BertConfig, BertTokenizer),
    "quant_bert": (QuantizedBertConfig, BertTokenizer),
}

MODEL_CLASS = {
    "bert": BertForSequenceClassification,
    "quant_bert": QuantizedBertForSequenceClassification,
}

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logger = logging.getLogger()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if set_type == 'test':
                label = None
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                label = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        else:
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = None
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return [None]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if set_type == 'test':
                label = None
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                if set_type == 'test':
                    text_a = line[1]
                    text_b = line[2]
                    label = None
                else:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_aug_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_aug.tsv")), "aug")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            if set_type == 'test':
                text_a = line[1]
                text_b = line[2]
                label = None
            else:
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def get_metric_key(task_name):
    if task_name == "cola":
        return "mcc"
    if task_name == "mrpc":
        return 'f1'
    if task_name == "sts-b":
        return 'pearson'
    if task_name == "qqp":
        return 'f1'
    return 'acc'


def get_train_steps_epochs(num_train_epochs, gradient_accumulation_steps, num_samples, batch_size=16):
    # return num_samples * num_train_epochs
    num = int(num_samples / batch_size if num_samples % batch_size == 0 else num_samples // batch_size + 1)
    return num // gradient_accumulation_steps * num_train_epochs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # noinspection PyBroadException
        try:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        except:
            label_id = 0

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
            logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        raise

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def do_eval(model, task_name, eval_dataloader, device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for _, batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
        else:
            raise

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss

    return result


def do_eval_v2(model, eval_dataloader, device, output_mode, eval_labels, num_labels):
    for _, batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _ = model(input_ids, segment_ids, input_mask)
        break  # one batch is enough for initialize activation alpha


def do_infer(model, eval_dataloader, device, output_mode, output_file, label_list):
    preds = []
    label_map = {i: label for i, label in enumerate(label_list)}
    for batch_ in tqdm(eval_dataloader, desc="Infering"):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            logits, _, _ = model(input_ids, segment_ids, input_mask)

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('index\tprediction\n')
        for i, pred in enumerate(preds):
            if output_mode == "classification":
                f.write(f'{i}\t{label_map[pred]}\n')
            else:
                f.write(f'{i}\t{pred}\n')


def main(task_name='', new_kd=False, print_log=True, data_dir='', model_dir=''):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=data_dir, type=str, )
    parser.add_argument("--model_dir", default=model_dir, type=str)
    parser.add_argument("--task_name", default=task_name, type=str)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument('--do_distill_with_quantization', default=1, type=int)
    parser.add_argument('--do_predict', default=0, type=int)
    parser.add_argument('--aug_train', default=0, type=int)
    parser.add_argument('--pred_distill', default=0, type=int)
    parser.add_argument('--do_quant', default=1, type=int)
    parser.add_argument("--quant_activation", default=1, type=int)
    args = parser.parse_args()
    if print_log:
        print(args)
    logger.info('The args: {}'.format(args))
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir, task_name)
    processed_data_dir = os.path.join(args.data_dir, 'preprocessed', task_name)

    args.student_model = os.path.join(args.model_dir, task_name)
    args.teacher_model = os.path.join(args.model_dir, task_name)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification"
    }

    # default_params = {
    #     "cola": {"max_seq_length": 64, "batch_size": 16, "eval_step": 50},
    #     "mnli": {"max_seq_length": 128, "batch_size": 32, "eval_step": 1000},
    #     "mrpc": {"max_seq_length": 128, "batch_size": 32, "eval_step": 200},
    #     "sst-2": {"max_seq_length": 64, "batch_size": 32, "eval_step": 200},
    #     "sts-b": {"max_seq_length": 128, "batch_size": 32, "eval_step": 50},
    #     "qqp": {"max_seq_length": 128, "batch_size": 32, "eval_step": 1000},
    #     "qnli": {"max_seq_length": 128, "batch_size": 32, "eval_step": 1000},
    #     "rte": {"max_seq_length": 128, "batch_size": 32, "eval_step": 100}
    # }

    default_params = {
        "cola": {"max_seq_length": 64, "batch_size": 16, "eval_step": 50},
        "mnli": {"max_seq_length": 128, "batch_size": 16, "eval_step": 100},
        "mrpc": {"max_seq_length": 128, "batch_size": 16, "eval_step": 50},
        "sst-2": {"max_seq_length": 64, "batch_size": 16, "eval_step": 50},
        "sts-b": {"max_seq_length": 128, "batch_size": 16, "eval_step": 50},
        "qqp": {"max_seq_length": 128, "batch_size": 16, "eval_step": 100},
        "qnli": {"max_seq_length": 128, "batch_size": 16, "eval_step": 100},
        "rte": {"max_seq_length": 128, "batch_size": 16, "eval_step": 50}
    }
    infer_files = {
        "cola": "CoLA.tsv",
        "mnli": "MNLI-m.tsv",
        "mrpc": "MRPC.tsv",
        "sst-2": "SST-2.tsv",
        "sts-b": "STS-B.tsv",
        "qqp": "QQP.tsv",
        "qnli": "QNLI.tsv",
        "rte": "RTE.tsv"
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Prepare seed
    set_seed(42)

    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        args.batch_size = int(args.batch_size * n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info("  num_labels = %d", num_labels)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=True)
    if args.do_distill_with_quantization:
        if args.aug_train:
            # noinspection PyBroadException
            try:
                train_file = os.path.join(processed_data_dir, 'aug_data')
                with open(train_file, 'rb') as f:
                    train_features = pickle.load(f)
            except:
                train_examples = processor.get_aug_examples(data_dir)
                train_features = convert_examples_to_features(train_examples, label_list,
                                                              args.max_seq_length, tokenizer, output_mode)
        else:
            # noinspection PyBroadException
            try:
                train_file = os.path.join(processed_data_dir, 'train_data')
                with open(train_file, 'rb') as f:
                    train_features = pickle.load(f)
            except:
                train_examples = processor.get_train_examples(data_dir)
                train_features = convert_examples_to_features(train_examples, label_list,
                                                              args.max_seq_length, tokenizer, output_mode)
                if not os.path.exists(processed_data_dir):
                    os.makedirs(processed_data_dir)
                train_file = os.path.join(processed_data_dir, 'train_data')
                with open(train_file, 'wb') as f:
                    pickle.dump(train_features, f)

        num_train_optimization_steps = int(len(train_features) / args.batch_size) * args.num_train_epochs
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        # noinspection PyBroadException
        try:
            dev_file = os.path.join(processed_data_dir, 'dev_data')
            with open(dev_file, 'rb') as f:
                eval_features = pickle.load(f)
        except:
            eval_examples = processor.get_dev_examples(data_dir)
            eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                         tokenizer, output_mode)
            if not os.path.exists(processed_data_dir):
                os.makedirs(processed_data_dir)
            dev_file = os.path.join(processed_data_dir, 'dev_data')
            with open(dev_file, 'wb') as f:
                pickle.dump(eval_features, f)

        eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
        if task_name == "mnli":
            processor = processors["mnli-mm"]()
            # noinspection PyBroadException
            try:
                dev_mm_file = os.path.join(processed_data_dir, 'dev-mm_data')
                with open(dev_mm_file, 'rb') as f:
                    mm_eval_features = pickle.load(f)
            except:
                mm_eval_examples = processor.get_dev_examples(data_dir)
                mm_eval_features = convert_examples_to_features(
                    mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
                if not os.path.exists(processed_data_dir):
                    os.makedirs(processed_data_dir)
                dev_mm_file = os.path.join(processed_data_dir, 'dev-mm_data')
                with open(dev_mm_file, 'wb') as f:
                    pickle.dump(mm_eval_features, f)

            mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)
            logger.info("  Num examples = %d", len(mm_eval_features))

            mm_eval_sampler = SequentialSampler(mm_eval_data)
            mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler, batch_size=args.batch_size)
        else:
            mm_eval_dataloader = 0.0
            mm_eval_labels = 0.0

        teacher_model = BertForSequenceClassification.from_pretrained(
            args.teacher_model,
            num_labels=num_labels,
            do_quantize=0,
            task_name=task_name
        )
        teacher_model.to(device)
        teacher_model.eval()
        student_model = QuantizedBertForSequenceClassification.from_pretrained(
            args.student_model,
            num_labels=num_labels,
            do_quantize=args.quant_activation,
            task_name=''
        )
        student_model.to(device)

        # ******************* initialize the wgt-alpha **************
        logger.info("***** wgt_alpha initialization *****")
        student_model.apply(kl_divergence_initilization)
        # ***********************************************************

        # ******************* initialize the act-alpha **************
        logger.info("***** activation alpha initialization *****")
        student_model.eval()
        student_model.apply(reset_act_alpha_true)
        do_eval_v2(student_model, eval_dataloader, device, output_mode, eval_labels, num_labels)
        student_model.apply(reset_act_alpha_false)
        # ***********************************************************

        result = do_eval(teacher_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        if task_name in acc_tasks:
            if task_name in ['sst-2', 'mnli', 'qnli', 'rte']:
                teacher_performance = f"acc:{result['acc']}"
                if task_name == "mnli":
                    result = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                                     device, output_mode, mm_eval_labels, num_labels)
                    teacher_performance += f"  mm-acc:{result['acc']}"
            elif task_name in ['mrpc', 'qqp']:
                teacher_performance = f"f1/acc:{result['f1']}/{result['acc']}"
            else:
                raise
        elif task_name in corr_tasks:
            teacher_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"
        elif task_name in mcc_tasks:
            teacher_performance = f"mcc:{result['mcc']}"
        else:
            raise
        teacher_performance = task_name + '   ' + teacher_performance

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if n_gpu > 1:
            student_model = torch.nn.DataParallel(student_model)
            teacher_model = torch.nn.DataParallel(teacher_model)

        # Prepare optimizer *****q8bert*****
        param_optimizer = list(student_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'LayerNorm.weight']
        alpha = ["wgt_alpha", "act_alpha"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in alpha)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in alpha)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if 'wgt_alpha' in n], 'lr': 1e-3, 'weight_decay': 2e-5},
            {'params': [p for n, p in param_optimizer if 'act_alpha' in n], 'lr': 3e-2, 'weight_decay': 1e-4},
        ]
        adam_epsilon = 1e-8
        warmup_steps = 0
        # total_steps = 0
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )
        # end optimizer
        loss_mse = MSELoss()

        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        global_step = 0
        best_dev_acc = 0.0
        previous_best = None

        total_start = time.time()
        for epoch_ in range(int(args.num_train_epochs)):
            for step, batch in enumerate(train_dataloader):
                start = time.time()
                student_model.train()
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits, teacher_logits)
                    if new_kd:
                        loss_fct = CrossEntropyLoss()
                        cls_loss += loss_fct(student_logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_mse = MSELoss()
                    cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                loss = cls_loss

                if not args.pred_distill:
                    teacher_layer_num = len(teacher_atts)
                    student_layer_num = len(student_atts)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                  teacher_att)
                        tmp_loss = loss_mse(student_att, teacher_att)
                        att_loss += tmp_loss

                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps
                    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                        rep_loss += tmp_loss

                    loss += (rep_loss + att_loss)

                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                if (global_step % args.eval_step == 0 or global_step == num_train_optimization_steps) and ((task_name in ['mnli', 'qnli', 'qqp'] and epoch_ == 2) or (task_name not in ['mnli', 'qnli', 'qqp'] and epoch_ >= 1)):
                    end = time.time()
                    logger.info("***** Running evaluation *****")
                    logger.info("  {} step of {} steps".format(global_step, num_train_optimization_steps))
                    if previous_best is not None:
                        logger.info(f"{teacher_performance}\nPrevious best = {previous_best}")

                    student_model.eval()

                    result = do_eval(student_model, task_name, eval_dataloader,
                                     device, output_mode, eval_labels, num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss

                    save_model = False

                    metric_key = get_metric_key(task_name)
                    if result[metric_key] > best_dev_acc:
                        best_dev_acc = result[metric_key]
                    if print_log:
                        print(global_step, str(best_dev_acc), str(result[metric_key]), end - start)

                    if save_model:
                        # Test mnli-mm
                        if task_name == "mnli":
                            result = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                             device, output_mode, mm_eval_labels, num_labels)
                            previous_best += f"mm-acc: {result['acc']}"
                        logger.info(teacher_performance)
                        logger.info(previous_best)
                        logger.info("******************** Save model ********************")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        # here we want to restore the quantized bert model #
                        model_to_save.save_pretrained(args.output_dir)

                        model_name = WEIGHTS_NAME
                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                    student_model.train()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
        total_end = time.time()
        student_model.eval()
        result = do_eval(student_model, task_name, eval_dataloader,
                         device, output_mode, eval_labels, num_labels)
        metric_key = get_metric_key(task_name)
        if result[metric_key] > best_dev_acc:
            best_dev_acc = result[metric_key]
        if print_log:
            print(global_step, str(best_dev_acc), str(result[metric_key]), total_end - total_start)
        return best_dev_acc

    if args.do_predict:
        processor = processors[task_name]()
        # noinspection PyBroadException
        try:
            test_file = os.path.join(processed_data_dir, 'test_data')
            with open(test_file, 'rb') as f:
                features = pickle.load(f)
        except:
            examples = processor.get_test_examples(data_dir)
            features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode)
            if not os.path.exists(processed_data_dir):
                os.makedirs(processed_data_dir)
            test_file = os.path.join(processed_data_dir, 'test_data')
            with open(test_file, 'wb') as f:
                pickle.dump(features, f)
        data, test_labels = get_tensor_data(output_mode, features)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
        output_file = os.path.join(args.output_dir, infer_files[task_name])

        model = QuantizedBertForSequenceClassification.from_pretrained(args.output_dir,
                                                                       num_labels=num_labels,
                                                                       from_8bit=True,
                                                                       do_quantize=args.quant_activation)
        model.to(device)
        model.eval()

        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(features))
        logger.info("  Batch size = %d", args.batch_size)

        do_infer(model, dataloader, device, output_mode, output_file, label_list)

        if task_name == "mnli":
            processor = processors["mnli-mm"]()
            # noinspection PyBroadException
            try:
                test_mm_file = os.path.join(processed_data_dir, 'test-mm_data')
                features = pickle.load(open(test_mm_file, 'rb'))
            except:
                examples = processor.get_test_examples(data_dir)
                features = convert_examples_to_features(
                    examples, label_list, args.max_seq_length, tokenizer, output_mode)
            data, labels = get_tensor_data(output_mode, features)

            logger.info("***** Running mm evaluation *****")
            logger.info("  Num examples = %d", len(features))

            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
            output_file = os.path.join(args.output_dir, 'MNLI-mm.tsv')
            do_infer(model, dataloader, device, output_mode, output_file, label_list)


if __name__ == "__main__":
    from config_bit import bit_config
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_dir = 'nlp_data/'
    model_dir = 'finetuned_teacher/'
    
    task_list = ['mnli', 'cola', 'mrpc', 'qnli', 'rte', 'sst-2', 'sts-b']
    bit_list = [3]
    kd_list = [False, True]
    for task in task_list:
        for bit in bit_list:
            for kd in kd_list:
                for key in bit_config:
                    if 'weight_bits' in bit_config[key]:
                        bit_config[key]['weight_bits'] = bit
                start_total = time.time()
                best_dev = main(task_name=task, new_kd=kd, print_log=True, data_dir=data_dir, model_dir=model_dir)
                end_total = time.time()
                print('task:', task,
                      'Best:', best_dev,
                      'bit', bit,
                      'new_kd', kd,
                      'time:', end_total - start_total)
