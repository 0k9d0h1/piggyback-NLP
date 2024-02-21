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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--datasets', type=str, default='s2orc',
                   help='Names of datasets')
FLAGS.add_argument('--model', type=str, default='bert-base-uncased',
                   help='type of pretrained model')
FLAGS.add_argument('--epoch', type=int, default=5)
FLAGS.add_argument('--seed', type=int, default=1)
args = FLAGS.parse_args()


class PiggybackStrategy(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, experiences: Any | Iterable, eval_streams: Sequence[Any | Iterable] | None = None, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return super().train(experiences, eval_streams, **kwargs)

    def forward(self):
        self.input_ids = self.mb_x[:, 0, :].type(torch.IntTensor).to('cuda')
        self.original_input_ids = self.mb_x[:, 1, :].type(torch.IntTensor).to(
            'cuda')
        self.token_type_ids = self.mb_x[:, 2, :].type(torch.IntTensor).to(
            'cuda')
        self.attention_mask = self.mb_x[:, 3, :].type(torch.IntTensor).to(
            'cuda')
        task_label = self.mb_task_id[0].item()
        return self.model(input_ids=self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask, task_label=task_label)

    def criterion(self):
        prediction_scores, seq_relationship_score = self.mb_output
        # labels = self.
        masked_token = self.input_ids.eq(103)
        loss_mask = self._criterion(
            prediction_scores[masked_token], self.original_input_ids[masked_token].type(torch.LongTensor).to('cuda'))
        loss_nsp = self._criterion(
            seq_relationship_score, self.mb_y.type(torch.LongTensor).to('cuda'))
        return loss_mask + loss_nsp

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


def copy_weights(model, model_pretrained):
    # Copy weights of pretrained model
    module_list = list(model_pretrained.modules())
    i = 0

    for module in model.modules():
        if i == len(module_list):
            break

        if 'ElementWiseLinear' in str(type(module)):
            module.weight.data.copy_(module_list[i].weight.data)
            module.bias.data.copy_(module_list[i].bias.data)

        elif 'Embedding' in str(type(module)) and 'Bert' not in str(type(module)):
            module.weight.data.copy_(module_list[i].weight.data)

        elif 'LayerNorm' in str(type(module)):
            module.weight.data.copy_(module_list[i].weight.data)
            module.bias.data.copy_(module_list[i].bias.data)
            module.eval()

        elif 'Dict' in str(type(module)) or 'BertForPreTraining' in str(type(module)):
            continue
        i += 1


random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model_pretrained = BertModel.from_pretrained(args.model)

if args.model == 'bert-base-uncased':
    config = BertConfig()
model = BertForPreTraining(config)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

interactive_logger = InteractiveLogger()
wandb_logger = WandBLogger(
    project_name="piggyback_ViT",
    run_name="piggyback_%s" % (args.datasets),
    path="../checkpoint",
    config=vars(args)
)
eval_plugin = EvaluationPlugin(
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
                                train_mb_size=16,
                                train_epochs=args.epoch,
                                eval_mb_size=16,
                                device='cuda',
                                evaluator=eval_plugin,
                                eval_every=1)

# benchmark_endtask = create_endtask_benchmark(args, tokenizer)
benchmark_unlabeled = create_unlabeled_benchmark(args, tokenizer)

# train_stream = benchmark_endtask.train_stream
# test_stream = benchmark_endtask.test_stream
train_stream = benchmark_unlabeled.train_stream
test_stream = benchmark_unlabeled.test_stream

copy_weights(model, model_pretrained)
torch.set_printoptions(profile="full")

results = []
for train_exp, test_exp in zip(train_stream, test_stream):
    print(len(train_exp.dataset))
    print("Start of experience: ", train_exp.current_experience)
    print("Current Classes: ", train_exp.classes_in_this_experience)

    cl_strategy.train(train_exp, eval_streams=[test_exp])
    print('Training completed')

    # check(model, model_pretrained)

    print('Computing accuracy on the test set')
    result = cl_strategy.eval(test_stream)

    results.append(result)
