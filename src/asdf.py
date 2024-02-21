import avalanche
import torch
import itertools
import pandas as pd
import random
import json
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
from dataset_small import create_small_benchmark
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Union
import multiprocessing
# torch.set_printoptions(profile="full")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--datasets', type=str, default='s2orc,realnews,pubmed',
                   help='Names of datasets')
FLAGS.add_argument('--model', type=str, default='bert-base-uncased',
                   help='type of pretrained model')
FLAGS.add_argument('--epoch', type=int, default=5)
FLAGS.add_argument('--seed', type=int, default=1)
args = FLAGS.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = BertForPreTraining(BertConfig())

# benchmark_endtask = create_endtask_benchmark(args, tokenizer)
# train_stream = benchmark_endtask.train_stream
# test_stream = benchmark_endtask.test_stream
# val_stream = benchmark_endtask.stream_factory("val", benchmark_endtask)
# for train_exp, test_exp, val_exp in zip(train_stream, test_stream, val_stream):
#     print(len(train_exp.dataset))
#     print(tokenizer.decode(
#         train_exp.dataset[0][0][0].type(torch.IntTensor).tolist()))

# benchmark_unlabeled = create_unlabeled_benchmark(args, tokenizer)
# train_stream = benchmark_unlabeled.train_stream
# test_stream = benchmark_unlabeled.test_stream
# val_stream = benchmark_unlabeled.stream_factory("val", benchmark_unlabeled)
# for train_exp, test_exp, val_exp in zip(train_stream, test_stream, val_stream):
#     print(len(train_exp.dataset))
#     print(train_exp.dataset[0])
#     print(test_exp.dataset[0])
#     print(val_exp.dataset[0])

# benchmark_small = create_small_benchmark(args, tokenizer)
# train_stream = benchmark_small.train_stream
# test_stream = benchmark_small.test_stream
# val_stream = benchmark_small.stream_factory("val", benchmark_small)
# for train_exp, test_exp, val_exp in zip(train_stream, test_stream, val_stream):
#     print(len(train_exp.dataset))
#     print(train_exp.dataset[0])

print(model)
