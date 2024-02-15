import avalanche
import torch
import itertools
import pandas as pd
import random
import jsonlines
import xml.etree.ElementTree as ET
import argparse
import torch.nn as nn
import avalanche.evaluation.metrics as amet

from avalanche.benchmarks.generators import tensors_benchmark, dataset_benchmark
from avalanche.training.templates import SupervisedTemplate
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from transformers import BertModel
from transformers import AutoTokenizer
from transformers import BertConfig
from modnets import BertForPreTraining
from datasets import Dataset
from torch.utils.data import TensorDataset
from dataset import create_unlabeled_benchmark, create_endtask_benchmark
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Union

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--datasets', type=str, default='s2orc',
                   help='Names of datasets')
FLAGS.add_argument('--model', type=str, default='bert-base-uncased',
                   help='type of pretrained model')
FLAGS.add_argument('--epoch', type=int, default=1)
args = FLAGS.parse_args()


class PiggybackStrategy(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, experiences: Any | Iterable, eval_streams: Sequence[Any | Iterable] | None = None, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return super().train(experiences, eval_streams, **kwargs)

    def forward(self):
        # input_ids = d[0, :]
        # token_type_ids = d[1, :]
        # attention_mask = d[2, :]
        input_ids = self.mb_x[:, 0, :].type(torch.IntTensor).to('cuda')
        token_type_ids = self.mb_x[:, 1, :].type(torch.IntTensor).to('cuda')
        attention_mask = self.mb_x[:, 2, :].type(torch.IntTensor).to('cuda')
        print(input_ids.shape, token_type_ids.shape,
              attention_mask.shape, self.mb_task_id)
        return self.model(input_ids, token_type_ids, attention_mask, self.mb_task_id)

    def backward(self):
        super().backward()

        for module in self.model.modules():
            if 'ElementWiseConv2d' in str(type(module)) or 'ElementWiseLinear' in str(type(module)):
                abs_weights = module.weight.data.abs()
                for i in range(len(module.masks)):
                    if module.masks[str(i)].grad is not None:
                        module.masks[str(i)].grad.data.div_(abs_weights.mean())

            elif 'ElementWiseMultiheadAttention' in str(type(module)):
                if not module._qkv_same_embed_dim:
                    abs_q_proj_weights = module.q_proj_weight.data.abs()
                    abs_k_proj_weights = module.k_proj_weight.data.abs()
                    abs_v_proj_weights = module.v_proj_weight.data.abs()
                    for i in range(len(module.masks)):
                        if module.masks[str(i)][0].grad is not None:
                            module.masks[str(i)][0].grad.data.div_(
                                abs_q_proj_weights.mean())
                            module.masks[str(i)][1].grad.data.div_(
                                abs_k_proj_weights.mean())
                            module.masks[str(i)][2].grad.data.div_(
                                abs_v_proj_weights.mean())
                else:
                    abs_in_proj_weights = module.in_proj_weight.data.abs()
                    for i in range(len(module.masks)):
                        if module.masks[str(i)].grad is not None:
                            module.masks[str(i)].grad.data.div_(
                                abs_in_proj_weights.mean())


tokenizer = AutoTokenizer.from_pretrained(args.model)
model_pretrained = BertModel.from_pretrained(args.model)

if args.model == 'bert-base-uncased':
    config = BertConfig()
model = BertForPreTraining(config)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

interactive_logger = InteractiveLogger()
# wandb_logger = WandBLogger(
#     project_name="piggyback_ViT",
#     run_name="piggyback_%s" % (args.datasets),
#     path="../checkpoint",
#     config=vars(args)
# )
eval_plugin = EvaluationPlugin(
    amet.accuracy_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    amet.loss_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    loggers=[interactive_logger]
)
cl_strategy = PiggybackStrategy(model,
                                optimizer,
                                criterion,
                                train_mb_size=128,
                                train_epochs=args.epoch,
                                eval_mb_size=128,
                                device='cuda',
                                evaluator=eval_plugin,
                                eval_every=1)

# benchmark_endtask = create_endtask_benchmark(args, tokenizer)
benchmark_unlabeled = create_unlabeled_benchmark(args, tokenizer)

# train_stream = benchmark_endtask.train_stream
# test_stream = benchmark_endtask.test_stream
train_stream = benchmark_unlabeled.train_stream
test_stream = benchmark_unlabeled.test_stream

torch.set_printoptions(profile="full")

results = []
for train_exp, test_exp in zip(train_stream, test_stream):
    # d = train_exp.dataset[0][0]
    # task_label = train_exp.dataset[0][2]
    # input_ids = d[0, :]
    # token_type_ids = d[1, :]
    # attention_mask = d[2, :]
    # a, b = model(input_ids=input_ids[None, :].type(torch.IntTensor), token_type_ids=token_type_ids[None, :].type(torch.IntTensor),
    #              attention_mask=attention_mask[None, :].type(torch.IntTensor), task_label=task_label)
    # print(a.shape, b.shape)
    # model.adaptation(train_exp)
    print(model)
    print("Start of experience: ", train_exp.current_experience)
    print("Current Classes: ", train_exp.classes_in_this_experience)

    cl_strategy.train(train_exp, eval_streams=[test_exp])
    print('Training completed')

    # check(model, model_pretrained)

    print('Computing accuracy on the test set')
    result = cl_strategy.eval(test_stream)

    results.append(result)
# print(model)


# self.mbatch
