import numpy as np
from transformers import T5ForConditionalGeneration
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils import AvalancheDataset
import avalanche.training.templates.base
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
import torch.nn
import avalanche
from transformers.utils import PaddingStrategy
from typing import Optional, Union, Any
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from avalanche.training.plugins import ReplayPlugin
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
import jsonlines
import json
import h5py
import torch
import itertools
import pandas as pd
import random
import xml.etree.ElementTree as ET
from datasets import load_dataset
from avalanche.benchmarks.generators import tensors_benchmark, dataset_benchmark
from transformers import BertModel
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import TensorDataset

MASK_PROP = 0.15


def preprocess_function_pubmed(pubmed):
    inputs = [x['MedlineCitation']['Article']
              ['Abstract']['AbstractText'] for x in pubmed['train']]


def mask_nsp_data(tokenizer, nsp_data):
    for i, tokens in enumerate(nsp_data):
        for j, token in enumerate(tokens['input_ids']):
            if token != 0 and token != 101 and token != 102:
                if random.random() < MASK_PROP:
                    if random.random() < 0.8:
                        nsp_data[i]['input_ids'][j] = 103
                    else:
                        if random.random() < 0.5:
                            nsp_data[i]['input_ids'][j] = random.randint(
                                1, len(tokenizer.get_vocab()))


def get_nsp_data(tokenizer, paragraphs):
    label = []
    nsp_data = []
    for paragraph in paragraphs:
        for i in range(len(paragraph) - 1):
            if random.random() < 0.5:
                label.append(1)
                nsp_data.append(
                    tokenizer(paragraph[i], paragraph[i+1], padding='max_length', max_length=128))
            else:
                label.append(0)
                nsp_data.append(
                    tokenizer(paragraph[i], random.choice(random.choice(paragraphs)), padding='max_length', max_length=128))
    return nsp_data, label


realnews = []
with jsonlines.open('../data/unlabeled/realnews/realnews.jsonl') as f:
    for i, lin in enumerate(f):
        sentences = lin['text'].strip().replace(
            '\n', '').lower().split('.')
        paragraph = []
        for sentence in sentences:
            if len(sentence) >= 2:
                paragraph.append(sentence)
        realnews.append(paragraph)
        if i == 10:
            break

# pubmed = load_dataset('pubmed', split='train[:10]')

# pm = []
# i = 0
# for entry in torch_iterable_dataset['train']:
#     pm.append(entry['MedlineCitation']['Article']['Abstract']['AbstractText'])
#     i += 1
#     if i == 1000:
#         break

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

nsp_data, label = get_nsp_data(tokenizer, realnews)
mask_nsp_data(tokenizer, nsp_data)

for a, n in zip(nsp_data, label):
    print(torch.Tensor(a['input_ids']))

# dss_realnews_train = []
# dss_realnews_test = []
# for i in range(0, 3):
#     masked = torch.Tensor(
#         tokenized_realnews['input_ids'][i * 3:(i+1)*3])
#     label = torch.Tensor(
#         [0, 0, 0])
#     dss_realnews_train.append(make_classification_dataset(
#         TensorDataset(masked, label), task_labels=i))
#     dss_realnews_test.append(make_classification_dataset(
#         TensorDataset(masked, label), task_labels=i))

# benchmark = dataset_benchmark(dss_realnews_train, dss_realnews_test)

# train_stream = benchmark.train_stream
# test_stream = benchmark.test_stream
