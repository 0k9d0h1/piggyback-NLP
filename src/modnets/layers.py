"""Contains novel layer definitions."""
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from torch.nn import init
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import avalanche.models as am
from avalanche.benchmarks.scenarios import CLExperience

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold=DEFAULT_THRESHOLD):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold=DEFAULT_THRESHOLD):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class PretrainingMultiTaskClassifier(am.MultiTaskModule):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias = bias
        self.initial_out_features = initial_out_features
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label

        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=self.initial_out_features, bias=self.bias)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x)


class MultiTaskClassifier(am.MultiTaskModule):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias = bias
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label
        curr_classes = len(experience.classes_in_this_experience)

        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=curr_classes, bias=self.bias)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x)


class ElementWiseMultiheadAttention(am.MultiTaskModule):
    """Modified multi-head attention with masks for weights."""
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.threshold_fn = threshold_fn

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Variable(torch.Tensor(
                embed_dim, embed_dim), requires_grad=False)
            self.k_proj_weight = Variable(torch.Tensor(
                embed_dim, self.kdim), requires_grad=False)
            self.v_proj_weight = Variable(torch.Tensor(
                embed_dim, self.vdim), requires_grad=False)
            self.q_proj_weight = self.q_proj_weight.to('cuda')
            self.k_proj_weight = self.k_proj_weight.to('cuda')
            self.v_proj_weight = self.v_proj_weight.to('cuda')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = Variable(torch.Tensor(
                3 * embed_dim, embed_dim), requires_grad=False)
            self.in_proj_weight = self.in_proj_weight.to('cuda')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = Variable(torch.Tensor(
                3 * embed_dim), requires_grad=False)
            self.in_proj_bias = self.in_proj_bias.to('cuda')
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = ElementWiseLinear(
            embed_dim, embed_dim, bias=bias,
            mask_init=mask_init, mask_scale=mask_scale,
            threshold_fn=threshold_fn, threshold=threshold)

        if add_bias_kv:
            self.bias_k = Variable(torch.Tensor(
                1, 1, embed_dim), requires_grad=False)
            self.bias_v = Variable(torch.Tensor(
                1, 1, embed_dim), requires_grad=False)
            self.bias_k = self.bias_k.to('cuda')
            self.bias_v = self.bias_v.to('cuda')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        # Initialize dictionary of masks
        self.masks = nn.ParameterDict({'0': self.make_elementwise_mask()})

        self.tmpn = 0

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def make_elementwise_mask(self):
        # Initialize real-valued mask weights.
        if not self._qkv_same_embed_dim:
            q_proj_mask_real = self.q_proj_weight.data.new(
                self.q_proj_weight.size())
            k_proj_mask_real = self.k_proj_weight.data.new(
                self.k_proj_weight.size())
            v_proj_mask_real = self.v_proj_weight.data.new(
                self.v_proj_weight.size())

            if self.mask_init == '1s':
                q_proj_mask_real.fill_(self.mask_scale)
                k_proj_mask_real.fill_(self.mask_scale)
                v_proj_mask_real.fill_(self.mask_scale)
            elif self.mask_init == 'uniform':
                q_proj_mask_real.uniform_(-1 *
                                          self.mask_scale, self.mask_scale)
                k_proj_mask_real.uniform_(-1 *
                                          self.mask_scale, self.mask_scale)
                v_proj_mask_real.uniform_(-1 *
                                          self.mask_scale, self.mask_scale)
            return Parameter(q_proj_mask_real), Parameter(k_proj_mask_real), Parameter(v_proj_mask_real)

        else:
            in_proj_mask_real = self.in_proj_weight.data.new(
                self.in_proj_weight.size())

            if self.mask_init == '1s':
                in_proj_mask_real.fill_(self.mask_scale)
            elif self.mask_init == 'uniform':
                in_proj_mask_real.uniform_(-1 *
                                           self.mask_scale, self.mask_scale)

            return Parameter(in_proj_mask_real)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label

        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_elementwise_mask()

    def forward(self, query: Tensor,
                key: Tensor,
                value: Tensor,
                task_label: int,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True,
                is_causal: bool = False) -> Tensor:

        return self.forward_single_task(query, key, value, task_label,
                                        key_padding_mask=key_padding_mask, need_weights=need_weights,
                                        attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)

    def forward_single_task(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            task_label: int,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

        # Apply threshold function and multiply to weight
        mask_real = self.masks[str(task_label)]
        if not self._qkv_same_embed_dim:
            if self.threshold_fn == 'binarizer':
                q_proj_mask_thresholded = Binarizer.apply(mask_real[0])
                k_proj_mask_thresholded = Binarizer.apply(mask_real[1])
                v_proj_mask_thresholded = Binarizer.apply(mask_real[2])
            elif self.threshold_fn == 'ternarizer':
                q_proj_mask_thresholded = Ternarizer.apply(mask_real[0])
                k_proj_mask_thresholded = Ternarizer.apply(mask_real[1])
                v_proj_mask_thresholded = Ternarizer.apply(mask_real[2])

            q_proj_weight_thresholded = q_proj_mask_thresholded * self.q_proj_weight
            k_proj_weight_thresholded = k_proj_mask_thresholded * self.k_proj_weight
            v_proj_weight_thresholded = v_proj_mask_thresholded * self.v_proj_weight

        else:
            if self.threshold_fn == 'binarizer':
                in_proj_mask_thresholded = Binarizer.apply(mask_real)
            elif self.threshold_fn == 'ternarizer':
                in_proj_mask_thresholded = Ternarizer.apply(mask_real)

            in_proj_weight_thresholded = in_proj_mask_thresholded * self.in_proj_weight

        out_proj_weight = self.out_proj.get_weight(task_label)
        out_proj_bias = self.out_proj.bias

        # self.tmpn += 1
        # if self.tmpn == 60:
        #     print('multihead', torch.sum(in_proj_mask_thresholded) /
        #           (self.in_proj_weight.shape[0] * self.in_proj_weight.shape[1]))
        #     self.tmpn = 0

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                in_proj_weight_thresholded,
                self.in_proj_bias,
                out_proj_weight,
                out_proj_bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(
                    attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        in_proj_weight_thresholded,
                        self.in_proj_bias,
                        out_proj_weight,
                        out_proj_bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0)
                                     for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight_thresholded, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight_thresholded, k_proj_weight=k_proj_weight_thresholded,
                v_proj_weight=v_proj_weight_thresholded,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight_thresholded, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(
                    batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type


class ElementWiseLinear(am.MultiTaskModule):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # Weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.masks = nn.ParameterDict({'0': self.make_mask()})
        self.tmpn = 0

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label

        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        # Get binarized/ternarized mask from real-valued mask.
        weight_thresholded = self.get_weight(task_label)
        # Get output using modified weight.
        return F.linear(x, weight_thresholded, self.bias)

    def make_mask(self):
        # Initialize real-valued mask weights.
        mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.
        return Parameter(mask_real)

    def get_weight(self, task_label):
        # For multi-head attention module
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.masks[str(task_label)])
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.masks[str(task_label)])

        weight_thresholded = mask_thresholded * self.weight

        return weight_thresholded

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


class ElementWiseEmbedding(am.MultiTaskModule):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        # Weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            num_embeddings, embedding_dim), requires_grad=False)
        self.masks = nn.ParameterDict({'0': self.make_mask()})

        self.sparse = sparse

    def make_mask(self):
        # Initialize real-valued mask weights.
        mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.
        return Parameter(mask_real)

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label

        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def get_weight(self, task_label):
        # For multi-head attention module
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.masks[str(task_label)])
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.masks[str(task_label)])

        weight_thresholded = mask_thresholded * self.weight

        return weight_thresholded

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        weight_thresholded = self.get_weight(task_label)
        return F.embedding(
            x, weight_thresholded, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
