import logging
import os
import sys

import torch
import math
from torch import nn
from transformer.modeling import (
    ACT2FN,
    BertAttention,
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertIntermediate,
    BertLayer,
    BertLayerNorm,
    BertModel,
    BertOutput,
    BertPooler,
    BertPreTrainedModel,
    BertSelfAttention,
    BertSelfOutput,
)

from quantization import (
    QuantizationConfig,
    QuantizedEmbedding,
    QuantizedLayer,
    QuantizedActivation,
    QuantizedLinear,
)

from config_bit import bit_config

logger = logging.getLogger(__name__)

QUANT_WEIGHTS_NAME = "quant_pytorch_model.bin"

QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bert-base-uncased":
        "https://nlp-architect-data.s3-us-west-2.amazonaws.com/models/transformers/bert-base-uncased.json",
    # noqa: E501
    "bert-large-uncased":
        "https://nlp-architect-data.s3-us-west-2.amazonaws.com/models/transformers/bert-large-uncased.json",
    # noqa: E501
}


def quantized_linear_setup(config, name, *args, **kwargs):
    """
    Get QuantizedLinear layer according to config params
    """
    try:
        quant_config = QuantizationConfig.from_dict(getattr(config, name))
        linear = QuantizedLinear.from_config(*args, **kwargs, config=quant_config)
    except AttributeError:
        linear = nn.Linear(*args, **kwargs)
    return linear


def quantized_embedding_setup(config, name, *args, **kwargs):
    """
    Get QuantizedEmbedding layer according to config params
    """
    try:
        quant_config = QuantizationConfig.from_dict(getattr(config, name))
        embedding = QuantizedEmbedding.from_config(*args, **kwargs, config=quant_config)
    except AttributeError:
        embedding = nn.Embedding(*args, **kwargs)
    return embedding


def quantized_activation_setup(config, name, *args, **kwargs):
    """
    Get QuantizedActivation layer according to config params
    """
    try:
        quant_config = QuantizationConfig.from_dict(getattr(config, name))
        activation = QuantizedActivation.from_config(*args, **kwargs, config=quant_config)
    except AttributeError:
        activation = None
    return activation


class QuantizedBertConfig(BertConfig):
    pretrained_config_archive_map = QUANT_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    @classmethod
    def update(cls, new_dict):
        for key, value in new_dict.items():
            cls.__dict__[key] = value


class QuantizedBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = quantized_embedding_setup(
            config, "word_embeddings", config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = quantized_embedding_setup(
            config, "position_embeddings", config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = quantized_embedding_setup(
            config, "token_type_embeddings", config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)  # config.layer_norm_eps
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertSelfAttention(BertSelfAttention):
    def __init__(self, config, name):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        # self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.do_quantize = config.do_quantize

        self.query = quantized_linear_setup(
            config, name + "attention_query", config.hidden_size, self.all_head_size
        )
        self.key = quantized_linear_setup(
            config, name + "attention_key", config.hidden_size, self.all_head_size
        )
        self.value = quantized_linear_setup(
            config, name + "attention_value", config.hidden_size, self.all_head_size
        )

        self.activation_key = quantized_activation_setup(config, name + "attention_key_layer")
        self.activation_query = quantized_activation_setup(config, name + "attention_query_layer")
        self.activation_value = quantized_activation_setup(config, name + "attention_value_layer")
        self.activation_att_prob = quantized_activation_setup(config, name + "attention_probs")

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask, output_att=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # where we did activation quantization#
        ########################################
        query_layer = self.activation_query(query_layer)
        key_layer = self.activation_key(key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # where we did activation quantization#
        ###############################################
        attention_probs = self.activation_att_prob(attention_probs)
        value_layer = self.activation_value(value_layer)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class QuantizedBertSelfOutput(BertSelfOutput):
    def __init__(self, config, name):
        super(BertSelfOutput, self).__init__()
        self.dense = quantized_linear_setup(
            config, name + "attention_output", config.hidden_size, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertAttention(BertAttention):
    def __init__(self, config, name):
        super(BertAttention, self).__init__()
        self.self = QuantizedBertSelfAttention(config, name)
        self.output = QuantizedBertSelfOutput(config, name)

    def prune_heads(self, heads):
        raise NotImplementedError("pruning heads is not implemented for Quantized BERT")


class QuantizedBertIntermediate(BertIntermediate):
    def __init__(self, config, name):
        super(BertIntermediate, self).__init__()
        self.dense = quantized_linear_setup(
            config, name + "intermediate", config.hidden_size, config.intermediate_size
        )
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, str)
        ):  # noqa: F821
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act


class QuantizedBertOutput(BertOutput):
    def __init__(self, config, name):
        super(BertOutput, self).__init__()
        self.dense = quantized_linear_setup(
            config, name + "output", config.intermediate_size, config.hidden_size
        )
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class QuantizedBertLayer(BertLayer):
    def __init__(self, config, name):
        super(BertLayer, self).__init__()
        self.attention = QuantizedBertAttention(config, name)
        self.intermediate = QuantizedBertIntermediate(config, name)
        self.output = QuantizedBertOutput(config, name)


class QuantizedBertEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [QuantizedBertLayer(config, name='layer_' + str(_) + '_') for _ in range(config.num_hidden_layers)]
        )


class QuantizedBertPooler(BertPooler):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = quantized_linear_setup(
            config, "pooler", config.hidden_size, config.hidden_size
        )
        self.activation = nn.Tanh()


class QuantizedBertPreTrainedModel(BertPreTrainedModel):
    config_class = BertConfig  # QuantizedBertConfig
    base_model_prefix = "quant_bert"

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, QuantizedLinear, QuantizedEmbedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, from_8bit=False, **kwargs):
        # from_pretrained(cls, pretrained_model_name_or_path, *args, from_8bit=False, **kwargs):
        """load trained model from 8bit model"""
        if not from_8bit:
            return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        config = kwargs.pop("config", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # Load config
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

        # Load model
        model_file = os.path.join(pretrained_model_name_or_path, QUANT_WEIGHTS_NAME)

        task_name = kwargs.get('task_name', None)
        if task_name is not None:
            kwargs.pop('task_name')
        config.__dict__['task_name'] = task_name

        # Instantiate model.
        model = cls(config)
        # model = cls(config, num_labels=2)  # we modified here for convienence
        ######################################################################

        # Set model to initialize variables to be loaded from quantized
        # checkpoint which are None by Default
        model.eval()
        # Get state dict of model
        state_dict = torch.load(model_file, map_location="cpu")
        logger.info("loading weights file {}".format(model_file))

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        if hasattr(model, "tie_weights"):
            model.tie_weights()  # make sure word embedding weights are still tied

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model

    def save_pretrained(self, save_directory):
        """save trained model in 8bit"""
        # super().save_pretrained(save_directory)
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        model_to_save.toggle_8bit(True)
        output_model_file = os.path.join(save_directory, QUANT_WEIGHTS_NAME)
        logger.info("Saving the 8 bit model!")
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.toggle_8bit(False)

    def toggle_8bit(self, mode: bool):
        def _toggle_8bit(module):
            if isinstance(module, QuantizedLayer):
                module.mode_8bit = mode

        self.apply(_toggle_8bit)
        if mode:
            training = self.training
            self.eval()
            self.train(training)


class QuantizedBertModel(QuantizedBertPreTrainedModel, BertModel):
    def __init__(self, config):
        # we only want BertForQuestionAnswering init to run to avoid unnecessary
        # initializations
        super(BertModel, self).__init__(config)
        quant_config = {"attention_query": {
            "mode": "ema"
        },
            "attention_key": {
                "mode": "ema"
            },
            "attention_value": {
                "mode": "ema",
                "requantize_output": False
            },
            "attention_output": {
                "mode": "ema",
                "requantize_output": False
            },
            "activation_query": {
                "mode": "ema",
                "requantize_output": False
            },
            "activation_key": {
                "mode": "ema",
                "requantize_output": False
            },
            "activation_value": {
                "mode": "ema",
                "requantize_output": False
            },
            "activation_att_prob": {
                "mode": "ema",
                "requantize_output": False
            },
            "ffn_intermediate": {
                "mode": "ema",
                "requantize_output": False
            },
            "ffn_output": {
                "mode": "ema",
                "requantize_output": False
            },
            "pooler": {
                "mode": "ema",
                "requantize_output": False
            },
            "head": {
                "mode": "ema",
                "requantize_output": False,
                "start_step": 200
            },
            "word_embeddings": {
                "mode": "ema"
            },
            "position_embeddings": {
                "mode": "ema"
            },
            "token_type_embeddings": {
                "mode": "none"
            }
        }
        config.update(quant_config)
        config.update(bit_config)
        # print(config)
        self.embeddings = QuantizedBertEmbeddings(config)
        self.encoder = QuantizedBertEncoder(config)
        self.pooler = QuantizedBertPooler(config)

        self.apply(self.init_weights)


class QuantizedBertForSequenceClassification(
    QuantizedBertPreTrainedModel, BertForSequenceClassification
):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = QuantizedBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)
        self.task_name = config.task_name


class QuantizedBertForQuestionAnswering(QuantizedBertPreTrainedModel, BertForQuestionAnswering):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = QuantizedBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)
