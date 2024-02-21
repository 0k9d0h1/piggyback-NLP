import random
import jsonlines
import torch
import os
import re

from tqdm import tqdm
from torch.utils.data import TensorDataset
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
from datasets import load_dataset

MASK_PROP = 0.15


def get_nsp_data(tokenizer, paragraphs):
    label = []
    nsp_data = []
    for paragraph in tqdm(paragraphs):
        for i in range(len(paragraph) - 1):
            if random.random() < 0.5:
                label.append(1)
                nsp_data.append(
                    tokenizer(paragraph[i], paragraph[i+1], padding='max_length', max_length=512, truncation='longest_first'))
            else:
                label.append(0)
                nsp_data.append(
                    tokenizer(paragraph[i], random.choice(random.choice(paragraphs)), padding='max_length', max_length=512, truncation='longest_first'))

    return nsp_data, label


def mask_nsp_data(tokenizer, paragraphs):
    nsp_data, label = get_nsp_data(tokenizer, paragraphs)
    original_input_ids = torch.stack(
        [torch.Tensor(a['input_ids']) for a in nsp_data])

    for i, tokens in tqdm(enumerate(nsp_data)):
        for j, token in enumerate(tokens['input_ids']):
            if token != 0 and token != 101 and token != 102:
                if random.random() < MASK_PROP:
                    if random.random() < 0.8:
                        nsp_data[i]['input_ids'][j] = 103
                    else:
                        if random.random() < 0.5:
                            r = random.randint(
                                1000, len(tokenizer.get_vocab()) - 1)
                            nsp_data[i]['input_ids'][j] = r
    return nsp_data, original_input_ids, label


def get_paragraphs(data):
    paragraphs = []
    if data == 'realnews':
        i = 0
        with jsonlines.open('../data/unlabeled/realnews/realnews.jsonl') as f:
            for lin in tqdm(f):
                sentences = lin['text'].strip().replace(
                    '\n', '').lower().split('. ')
                paragraph = []
                for sentence in sentences:
                    if len(sentence) >= 2:
                        paragraph.append(sentence)
                        i += 1
                if len(paragraph) != 0:
                    paragraphs.append(paragraph)

                if i > 100:
                    break

    elif data == 'pubmed':
        pubmed = load_dataset('pubmed', streaming=True, trust_remote_code=True)
        i = 0
        for entry in tqdm(pubmed['train']):
            abstracttext = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
            sentences = abstracttext.strip().replace('\n', '').lower().split('. ')
            paragraph = []
            for sentence in sentences:
                if len(sentence) >= 2:
                    paragraph.append(sentence)
                    i += 1
            if len(paragraph) != 0:
                paragraphs.append(paragraph)

            if i > 100:
                break

    elif data == 's2orc':
        file_list = os.listdir('../data/unlabeled/s2orc')
        for file in tqdm(file_list):
            with jsonlines.open('../data/unlabeled/s2orc/%s' % file) as f:
                i = 0
                for lin in f:
                    if lin['openaccessinfo']['externalids']['ACL'] is not None:
                        sentences = lin['abstract'].strip().replace(
                            '\n', '').lower().split('. ')
                        paragraph = []
                        for sentence in sentences:
                            if len(sentence) >= 2:
                                paragraph.append(sentence)
                                i += 1
                        if len(paragraph) != 0:
                            paragraphs.append(paragraph)
            if i > 100:
                break
    return paragraphs


def create_small_benchmark(args, tokenizer):
    dat = args.datasets.split(',')
    train_datasets = []
    test_datasets = []
    val_datasets = []

    for task_label, dataset in enumerate(dat):
        path = "../datasets/%s/small" % (dataset)
        dir = os.listdir(path)
        if len(dir) == 0:
            paragraphs = get_paragraphs(dataset)
            nsp_data, original_input_ids, label = mask_nsp_data(
                tokenizer, paragraphs)

            input_ids = torch.stack(
                [torch.Tensor(a['input_ids']).type(torch.IntTensor) for a in nsp_data])
            token_type_ids = torch.stack(
                [torch.Tensor(a['token_type_ids']).type(torch.IntTensor) for a in nsp_data])
            attention_mask = torch.stack(
                [torch.Tensor(a['attention_mask']).type(torch.IntTensor) for a in nsp_data])
            label = torch.Tensor(label)
            length = label.shape[0]

            nsp_data = torch.stack(
                [input_ids, original_input_ids, token_type_ids, attention_mask], dim=1)

            idx = torch.randperm(length)
            nsp_data = nsp_data[idx].view(nsp_data.size())
            label = label[idx].view(label.size())

            train_dataset = TensorDataset(
                nsp_data[:int(length * 4 / 5)], label[:int(length * 4 / 5)])
            val_dataset = TensorDataset(nsp_data[int(
                length * 4 / 5):int(length * 9 / 10)], label[int(length * 4 / 5):int(length * 9 / 10)])
            test_dataset = TensorDataset(
                nsp_data[int(length * 9 / 10):], label[int(length * 9 / 10):])

            torch.save(train_dataset, "%s/train.pt" % (path))
            torch.save(val_dataset, "%s/val.pt" % (path))
            torch.save(test_dataset, "%s/test.pt" % (path))

        else:
            train_dataset = torch.load("%s/train.pt" % (path))
            val_dataset = torch.load("%s/val.pt" % (path))
            test_dataset = torch.load("%s/test.pt" % (path))

        train_dataset_with_task_label = make_classification_dataset(
            train_dataset, task_labels=task_label)
        val_dataset_with_task_label = make_classification_dataset(
            val_dataset, task_labels=task_label)
        test_dataset_with_task_label = make_classification_dataset(
            test_dataset, task_labels=task_label)

        train_datasets.append(train_dataset_with_task_label)
        val_datasets.append(val_dataset_with_task_label)
        test_datasets.append(test_dataset_with_task_label)

    return dataset_benchmark(train_datasets, test_datasets, other_streams_datasets={"val": val_datasets})
