import torch
import torch.nn as nn
import avalanche.models as am
import torch.nn.functional as F

from torch.autograd import Variable
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import Optional
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
        self.bias_ = bias
        self.initial_out_features = initial_out_features
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, experience: CLExperience):
        super().adaptation(experience)
        task_label = experience.task_label

        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=self.initial_out_features, bias=self.bias_)

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
        self.tmpn += 1
        if self.tmpn == 1000:
            print('linear', torch.sum(self.mask_thresholded) /
                  (self.weight.shape[0] * self.weight.shape[1]))
            self.tmpn = 0
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
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)])
        elif self.threshold_fn == 'ternarizer':
            self.mask_thresholded = Ternarizer.apply(
                self.masks[str(task_label)])

        weight_thresholded = self.mask_thresholded * self.weight

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
