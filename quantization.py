# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# pylint: disable=no-member
"""
Quantization ops
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from enum import Enum, auto
import logging
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from config import Config


logger = logging.getLogger(__name__)

def weight_quantization(b):

    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input = input.div(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)               # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply

def act_quantization(b):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(min=-1, max=1)
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign*i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)
            return grad_input, grad_alpha

    return _uq().apply


def get_dynamic_scale(x, bits, with_grad=False):
    """Calculate dynamic scale for quantization from input by taking the
    maximum absolute value from x and number of bits"""
    with torch.set_grad_enabled(with_grad):
        threshold = x.abs().max()
    return get_scale(bits, threshold)


def get_scale(bits, threshold):
    """Calculate scale for quantization according to some constant and number of bits"""
    return calc_max_quant_value(bits) / threshold


def calc_max_quant_value(bits):
    """Calculate the maximum symmetric quantized value according to number of bits"""
    return 2 ** (bits - 1) - 1


def quantize(input, scale, bits):
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = calc_max_quant_value(bits)
    return input.mul(scale).round().clamp(-thresh, thresh)


def dequantize(input, scale):
    """linear dequantization according to some scale"""
    return input.div(scale)


# TODO(ofir) future work, implement a layer that uses this function that gives a more comfortable
class FakeLinearQuantizationWithSTE(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop"""

    @staticmethod
    def forward(ctx, input, scale, bits=8):
        """fake quantize input according to scale and number of bits, dequantize
        quantize(input))"""
        return dequantize(quantize(input, scale, bits), scale)

    @staticmethod
    def backward(ctx, grad_output):
        """Calculate estimated gradients for fake quantization using
        Straight-Through Estimator (STE) according to:
        https://openreview.net/pdf?id=B1ae1lZRb"""
        return grad_output, None, None


class QuantizationMode(Enum):
    NONE = auto()
    DYNAMIC = auto()
    EMA = auto()


_fake_quantize = FakeLinearQuantizationWithSTE.apply


class QuantizedLayer(ABC):
    """Quantized Layer interface"""

    CONFIG_ATTRIBUTES = ["weight_bits", "start_step", "mode"]
    REPR_ATTRIBUTES = ["mode", "weight_bits"]

    def __init__(self, *args, weight_bits=8, start_step=0, mode="none", **kwargs):
        if weight_bits < 2:
            raise ValueError(f"weight_bits={weight_bits} must be higher than 1 ")
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.mode = QuantizationMode[mode.upper()]
        self.start_step = start_step
        self.register_buffer("_step", torch.zeros(1))
        # buffers for inference
        self.register_buffer("quantized_weight", None)
        self.register_buffer("_weight_scale", None)
        # handle import and export in 8bit
        self.mode_8bit = False
        self._imported_from_quantized = False
        # register saving hook
        self._register_state_dict_hook(self._state_dict_hook)
        # a flag for activation initilization
        self.act_init_flag = False

    def forward(self, input):
        # if self.mode == QuantizationMode.NONE:
        #     return super().forward(input)
        # if self.training:
        #     if self._step >= self.start_step:
        #         out = self.training_quantized_forward(input)
        #     else:
        #         out = super().forward(input)
        #     self._step += 1
        # else:
        #     out = self.inference_quantized_forward(input)
        if self.mode == QuantizationMode.NONE:
            return super().forward(input)
        else:   
            return self.training_quantized_forward(input)
        # return out

    @abstractmethod
    def training_quantized_forward(self, input):
        """Implement forward method to be used while training"""

    @abstractmethod
    def inference_quantized_forward(self, input):
        """Implement forward method to be used while evaluating"""

    @classmethod
    def from_config(cls, *args, config=None, **kwargs):
        """Initialize quantized layer from config"""
        return cls(*args, **kwargs, **{k: getattr(config, k) for k in cls.CONFIG_ATTRIBUTES})

    @property
    def fake_quantized_weight(self):
        return _fake_quantize(self.weight, self.weight_scale, self.weight_bits)

    @property
    def weight_scale(self):
        return (
            get_dynamic_scale(self.weight, self.weight_bits)
            if self.training
            else self._weight_scale
        )

    def train(self, mode=True):
        """handle transition between quantized model and simulated quantization"""
        if self.training != mode:
            if mode:
                if self._imported_from_quantized:
                    raise RuntimeError(
                        "Model imported from quantized checkpoint cannot be moved to \
                            training mode"
                    )
                self._train()
            else:
                self._eval()
        super().train(mode)

    def _train(self):
        """function to be called by self.train(mode=True) which modifies modules attributes\
             according to the model"""

    def _eval(self):
        """function to be called by self.train(mode=False), or eval() which modifies modules\
             attributes according to the model"""
        self._weight_scale = self.weight_scale
        self.quantized_weight = quantize(self.weight, self.weight_scale, self.weight_bits)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """check if model is loaded from quantized checkpoint or regular checkpoint"""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if state_dict.get(prefix + "quantized_weight", None) is not None:
            if self.training:
                raise RuntimeError(
                    "Can't load quantized model in training mode, first change model's \
                         to evaluation and then load the saved model"
                )
            self._imported_from_quantized = True

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        """hook to be registered to module when exporting the model to 8bit, can be overrided\
             to customize to layer behaviour"""
        if module.mode_8bit and module.mode != QuantizationMode.NONE:
            state_dict.pop(prefix + "weight", None)
            state_dict.pop(prefix + "_step", None)
            state_dict[prefix + "quantized_weight"] = state_dict[prefix + "quantized_weight"].char()
        else:
            state_dict.pop(prefix + "quantized_weight", None)
            state_dict.pop(prefix + "_weight_scale", None)
        # state_dict.pop(prefix + "weight", None)
        # state_dict.pop(prefix + "_step", None)
        # state_dict[prefix + "quantized_weight"] = state_dict[prefix + "quantized_weight"].char()

    def extra_repr(self):
        s = ""
        for entry in self.REPR_ATTRIBUTES:
            s += f", {entry}={getattr(self, entry)}"
        return super().extra_repr() + s


class QuantizedLinear(QuantizedLayer, nn.Linear):
    """Linear layer with quantization aware training capability"""

    CONFIG_ATTRIBUTES = QuantizedLayer.CONFIG_ATTRIBUTES + [
        "activation_bits",
        "requantize_output",
        "ema_decay",
    ]
    REPR_ATTRIBUTES = QuantizedLayer.REPR_ATTRIBUTES + [
        "activation_bits",
        "accumulation_bits",
        "ema_decay",
        "requantize_output",
    ]

    def __init__(
        self, *args, activation_bits=8, requantize_output=True, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if activation_bits < 2:
            raise ValueError(f"activation_bits={activation_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.requantize_output = requantize_output
        self.register_buffer("input_thresh", torch.zeros(1))
        if self.requantize_output:
            self.register_buffer("output_thresh", torch.zeros(1))
        # real quantization
        if kwargs.get("bias", True):
            self.register_buffer("_quantized_bias", None)
            self.register_buffer("bias_scale", None)
        
        # self.weight_bits = 8.0
        self.weight_quant = weight_quantization(b=self.weight_bits - 1.0)
        deviceId = torch.cuda.current_device()
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32, device=torch.device('cuda:{}'.format(deviceId))))
        self.act_quant = act_quantization(b=self.activation_bits - 1.0)
        self.act_alpha = torch.nn.Parameter(torch.tensor(16.0, dtype=torch.float32, device=torch.device('cuda:{}'.format(deviceId))))

    def reset_wgt_alpha(self):
        scaled_weight = self.weight.mul(1.0).reshape([-1, 1, 1, 1])

        # dim = scaled_weight.dim()
        num_elem = scaled_weight.nelement()
        numpy_weight = scaled_weight.detach().cpu().reshape(-1).numpy()
        sorted_weight = np.sort(numpy_weight)
        min_index = np.int(np.around(num_elem * 0.025))
        max_index = num_elem - min_index
        result = 0.0
        if np.abs(sorted_weight[min_index]) >= np.abs(sorted_weight[max_index]):
           result = np.abs(sorted_weight[min_index])
        else:
            result = np.abs(sorted_weight[max_index])
        self.wgt_alpha.data.fill_(result * 1.0)
        return self

    def kl_divergence_activation(self, input):
        scaled_weight = input.mul(1.0).reshape([-1, 1, 1, 1])
        scaled_weight_npy = scaled_weight.detach().cpu().numpy()
        bitwidth = self.activation_bits
        mn = 0
        mx = np.abs(scaled_weight_npy).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(scaled_weight_npy), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def kl_divergence_initilization(self):
        scaled_weight = self.weight.mul(1.0).reshape([-1, 1, 1, 1])
        scaled_weight_npy = scaled_weight.detach().cpu().numpy()
        bitwidth = self.weight_bits
        mn = 0
        mx = np.abs(scaled_weight_npy).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(scaled_weight_npy), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        self.wgt_alpha.data.fill_(threshold * 1.0)
        return self

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        # assert self.training, "should only be called when training"
        # if self.mode == QuantizationMode.EMA:
        #     self._update_ema(self.input_thresh, input.detach())
        # input_scale = self._get_input_scale(input)
        # out = F.linear(
        #     _fake_quantize(input, input_scale, self.activation_bits),
        #     self.fake_quantized_weight,
        #     self.bias,
        # )
        # if self.requantize_output:
        #     if self.mode == QuantizationMode.EMA:
        #         self._update_ema(self.output_thresh, out.detach())
        #     out = _fake_quantize(out, self._get_output_scale(out), self.activation_bits)
        if not self.act_init_flag:
            #  weight = self.weight.mul(1.0)
            #  weight_q = self.weight_quant(weight, self.wgt_alpha)
             weight_q = self.weight_quant(self.weight, self.wgt_alpha)
             input_q = self.act_quant(input, self.act_alpha)
             return F.linear(
              input_q,
              weight_q,
              self.bias,
             )
        else:
            #  weight = self.weight.mul(1.0)
            #  weight_q = self.weight_quant(weight, self.wgt_alpha)
             weight_q = self.weight_quant(self.weight, self.wgt_alpha)
             threshold = self.kl_divergence_activation(input)
             self.act_alpha.data.fill_(threshold * 1.0)
             print("activation alpha_1: ", self.act_alpha)
             input_q = self.act_quant(input, self.act_alpha)
             return F.linear(
              input_q,
              weight_q,
              self.bias,
             )


    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        self.bias_scale = self.weight_scale * input_scale
        quantized_input = quantize(input, input_scale, self.activation_bits)
        out = F.linear(quantized_input, self.quantized_weight, self.quantized_bias)
        # TODO(ofir) fuse the operation of requantization with dequantiz
        out = dequantize(out, self.bias_scale)
        if self.requantize_output:
            output_scale = self._get_output_scale(out)
            out = dequantize(quantize(out, output_scale, self.activation_bits), output_scale)
        return out

    def _eval(self):
        super()._eval()
        if self.mode == QuantizationMode.EMA and self.bias is not None:
            self.bias_scale = self._get_input_scale() * self.weight_scale
            self.quantized_bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        """hook to be registered to module when exporting the model to 8bit,\
             can be overrided to customize to layer behaviour"""
        super()._state_dict_hook(module, state_dict, prefix, local_metadata)
        if module.mode_8bit:
            if module.mode == QuantizationMode.EMA:
                state_dict.pop(prefix + "bias", None)
                try:
                    state_dict[prefix + "_quantized_bias"] = state_dict[
                        prefix + "_quantized_bias"
                    ].int()
                except KeyError:
                    # in case there is no bias dont do anything
                    pass
        else:
            state_dict.pop(prefix + "_quantized_bias", None)
            state_dict.pop(prefix + "bias_scale", None)

    @property
    def quantized_bias(self):
        try:
            if self.mode == QuantizationMode.EMA:
                bias = self._quantized_bias
            elif self.mode == QuantizationMode.DYNAMIC:
                bias = quantize(self.bias, self.bias_scale, self.accumulation_bits)
            else:
                raise RuntimeError(f"Unknown quantization mode: {self.mode}")
        except AttributeError:
            bias = None
        return bias

    @quantized_bias.setter
    def quantized_bias(self, value):
        self._quantized_bias = value

    def _get_input_scale(self, input=None):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_output_scale(self, output=None):
        return self._get_activation_scale(output, self.output_thresh)

    def _get_activation_scale(self, activation, threshold):
        if self.mode == QuantizationMode.DYNAMIC:
            scale = get_dynamic_scale(activation, self.activation_bits)
        elif self.mode == QuantizationMode.EMA:
            scale = get_scale(self.activation_bits, threshold)
        return scale

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        assert self._step >= self.start_step
        if self._step == self.start_step:
            ema.fill_(reduce_fn(input))
        else:
            ema.sub_((1 - self.ema_decay) * (ema - reduce_fn(input)))


class QuantizedEmbedding(QuantizedLayer, nn.Embedding):
    """Embedding layer with quantization aware training capability"""
    def __init__(
            self, *args, activation_bits=8, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_quant = weight_quantization(b=self.weight_bits - 1.0)
        deviceId = torch.cuda.current_device()
        self.wgt_alpha = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32, device=torch.device('cuda:{}'.format(deviceId))))

    def training_quantized_forward(self, input):
        """Return quantized embeddings"""
        # assert self.training, "should only be called when training"
        # return F.embedding(
        #     input,
        #     self.fake_quantized_weight,
        #     self.padding_idx,
        #     self.max_norm,
        #     self.norm_type,
        #     self.scale_grad_by_freq,
        #     self.sparse,
        # )
        # weight = self.weight.mul(1.0)
        # weight_q = self.weight_quant(weight, self.wgt_alpha)
        weight_q = self.weight_quant(self.weight, self.wgt_alpha)
        return F.embedding(
            input,
            weight_q,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def inference_quantized_forward(self, input):
        """forward to be used during inference"""
        assert not self.training, "should only be called when not training"
        q_embeddings = F.embedding(
            input,
            self.quantized_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return dequantize(q_embeddings, self.weight_scale)

    def kl_divergence_initilization(self):
        scaled_weight = self.weight.mul(1.0).reshape([-1, 1, 1, 1])
        scaled_weight_npy = scaled_weight.detach().cpu().numpy()
        bitwidth = self.weight_bits
        mn = 0
        mx = np.abs(scaled_weight_npy).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(scaled_weight_npy), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        self.wgt_alpha.data.fill_(threshold * 1.0)
        return self


class QuantizedActivation(QuantizedLayer, nn.Module):
    """Activation layer with quantization aware training capability"""
    CONFIG_ATTRIBUTES = QuantizedLayer.CONFIG_ATTRIBUTES + [
        "activation_bits",
        "ema_decay",
    ]
    REPR_ATTRIBUTES = QuantizedLayer.REPR_ATTRIBUTES + [
        "activation_bits",
        "accumulation_bits",
        "ema_decay"
    ]

    def __init__(
            self, *args, activation_bits=8, ema_decay=0.9999, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if activation_bits < 2:
            raise ValueError(f"activation_bits={activation_bits} must be higher than 1 ")
        self.activation_bits = activation_bits
        self.accumulation_bits = 32
        self.ema_decay = ema_decay
        self.register_buffer("input_thresh", torch.zeros(1))
        deviceId = torch.cuda.current_device()
        self.act_quant = act_quantization(b=self.activation_bits - 1.0)
        self.act_alpha = torch.nn.Parameter(torch.tensor(16.0, dtype=torch.float32, device=torch.device('cuda:{}'.format(deviceId))))

    def kl_divergence_activation(self, input):
        scaled_weight = input.mul(1.0).reshape([-1, 1, 1, 1])
        scaled_weight_npy = scaled_weight.detach().cpu().numpy()
        bitwidth = self.activation_bits
        mn = 0
        mx = np.abs(scaled_weight_npy).max()
        if np.isclose(mx, 0.0):
            return 0.0
        hist, bin_edges = np.histogram(np.abs(scaled_weight_npy), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def training_quantized_forward(self, input):
        """Return quantized activation"""
        # assert self.training, "should only be called when training"
        # if self.mode == QuantizationMode.EMA:
        #     self._update_ema(self.input_thresh, input.detach())
        # input_scale = self._get_input_scale(input)
        # activation_out = _fake_quantize(input, input_scale, self.activation_bits)
        if not self.act_init_flag:
            return self.act_quant(input, self.act_alpha)
        else: 
            threshold = self.kl_divergence_activation(input)
            self.act_alpha.data.fill_(threshold * 1.0)
            print("activation alpha_2: ", self.act_alpha)
            return self.act_quant(input, self.act_alpha)       

    def inference_quantized_forward(self, input):
        """forward to be used during inference"""
        assert not self.training, "should only be called when not training"
        input_scale = self._get_input_scale(input)
        quantized_input = quantize(input, input_scale, self.activation_bits)
        return dequantize(quantized_input, input_scale)

    def _get_input_scale(self, input=None):
        return self._get_activation_scale(input, self.input_thresh)

    def _get_activation_scale(self, activation, threshold):
        if self.mode == QuantizationMode.DYNAMIC:
            scale = get_dynamic_scale(activation, self.activation_bits)
        elif self.mode == QuantizationMode.EMA:
            scale = get_scale(self.activation_bits, threshold)
        else:
            scale = 1.0
        return scale

    def _eval(self):
        pass

    @staticmethod
    def _state_dict_hook(module, state_dict, prefix, local_metadata):
        pass

    def _update_ema(self, ema, input, reduce_fn=lambda x: x.abs().max()):
        """Update exponential moving average (EMA) of activations thresholds.
        the reduce_fn calculates the current threshold from the input tensor"""
        assert self._step >= self.start_step
        if self._step == self.start_step:
            ema.fill_(reduce_fn(input))
        else:
            ema.sub_((1 - self.ema_decay) * (ema - reduce_fn(input)))

# we use this function before training
def reset_wgt_alpha(mod):
    if type(mod) in set([QuantizedLinear]):
       mod.reset_wgt_alpha()
# we use this function before training
def kl_divergence_initilization(mod):
    if type(mod) in set([QuantizedLinear, QuantizedEmbedding]):
       mod.kl_divergence_initilization()
# we use this function when doing inference before training
def reset_act_alpha_true(mod):
    if type(mod) in set([QuantizedLinear, QuantizedActivation]):
       mod.act_init_flag = True

def reset_act_alpha_false(mod):
    if type(mod) in set([QuantizedLinear, QuantizedActivation]):
       mod.act_init_flag = False

class QuantizationConfig(Config):
    """Quantization Configuration Object"""

    ATTRIBUTES = {
        "activation_bits": 8,
        "weight_bits":8,
        "mode": "none",
        "start_step": 0,
        "ema_decay": 0.9999,
        "requantize_output": True,
    }
