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
    for paragraph in paragraphs:
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
    return nsp_data, label


def get_paragraphs(data):
    paragraphs = []
    if data == 'realnews':
        with jsonlines.open('../data/unlabeled/realnews/realnews.jsonl') as f:
            for i, lin in enumerate(f):
                sentences = lin['text'].strip().replace(
                    '\n', '').lower().split('. ')
                paragraph = []
                for sentence in sentences:
                    if len(sentence) >= 2:
                        paragraph.append(sentence)
                if len(paragraph) != 0:
                    paragraphs.append(paragraph)
                if i == 2:
                    break

    elif data == 'pubmed':
        pubmed = load_dataset('pubmed', streaming=True, trust_remote_code=True)
        i = 0
        for entry in pubmed['train']:
            abstracttext = entry['MedlineCitation']['Article']['Abstract']['AbstractText']
            sentences = abstracttext.strip().replace('\n', '').lower().split('. ')
            paragraph = []
            for sentence in sentences:
                if len(sentence) >= 2:
                    paragraph.append(sentence)
                    i += 1
            if len(paragraph) != 0:
                paragraphs.append(paragraph)
            if i > 10:
                break

    elif data == 's2orc':
        file_list = os.listdir('../data/unlabeled/s2orc')
        for file in file_list:
            with jsonlines.open('../data/unlabeled/s2orc/%s' % file) as f:
                i = 0
                for lin in f:
                    if lin['openaccessinfo']['externalids']['ACL'] is not None:
                        i += 1
                        sentences = lin['abstract'].strip().replace(
                            '\n', '').lower().split('. ')
                        paragraph = []
                        for sentence in sentences:
                            if len(sentence) >= 2:
                                paragraph.append(sentence)
                        if len(paragraph) != 0:
                            paragraphs.append(paragraph)
                        if i == 2:
                            break
    return paragraphs


def create_unlabeled_benchmark(args, tokenizer):
    dat = args.datasets.split(',')
    train_datasets = []
    test_datasets = []

    for task_label, dataset in enumerate(dat):
        paragraphs = get_paragraphs(dataset)
        nsp_data, label = mask_nsp_data(tokenizer, paragraphs)

        input_ids = torch.stack(
            [torch.Tensor(a['input_ids']) for a in nsp_data])
        token_type_ids = torch.stack(
            [torch.Tensor(a['token_type_ids']) for a in nsp_data])
        attention_mask = torch.stack(
            [torch.Tensor(a['attention_mask']) for a in nsp_data])
        label = torch.Tensor(label)
        length = label.shape[0]

        nsp_data = torch.stack(
            [input_ids, token_type_ids, attention_mask], dim=1)

        idx = torch.randperm(length)
        nsp_data = nsp_data[idx].view(nsp_data.size())
        label = label[idx].view(label.size())

        train_datasets.append(make_classification_dataset(
            TensorDataset(nsp_data[:int(length * 4 / 5)], label[:int(length * 4 / 5)]), task_labels=task_label))
        test_datasets.append(make_classification_dataset(
            TensorDataset(nsp_data[int(length * 4 / 5):], label[int(length * 4 / 5):]), task_labels=task_label))

    return dataset_benchmark(train_datasets, test_datasets)


def create_endtask_benchmark(args, tokenizer):
    dat = args.datasets.split(',')
    train_datasets = []
    test_datasets = []

    for task_label, dataset in enumerate(dat):
        if dataset == 'realnews':
            hyperpartisan = load_dataset('hyperpartisan_news_detection', 'bypublisher',
                                         trust_remote_code=True)
            str_to_idx = {'false': 0,
                          'true': 1}

            label_train = []
            input_train = []
            for i, lin in enumerate(hyperpartisan['train']):
                text = re.sub('<[^>]+>', '', lin['text'])
                index = text.find('&#')
                text = text.replace("..", "")
                while index > -1:
                    idx_semicolon = index + 2
                    if index + 2 < len(text) and text[index+2].isdigit():
                        while idx_semicolon < len(text) and text[idx_semicolon] != ';':
                            idx_semicolon += 1
                            if index - idx_semicolon > 6:
                                break
                        if text[index+2:idx_semicolon].isdigit():
                            unicode_decimal = int(text[index+2:idx_semicolon])
                            unicode = f'&#%d;' % unicode_decimal
                            text = text.replace(unicode,
                                                chr(unicode_decimal))
                    index = text.find('&#')
                text = text.replace("&amp;#160", " ")
                text = text.replace("&amp;amp;", "&")
                text = text.replace("&amp;gt;", ">")

                label_train.append(int(lin['hyperpartisan']))

                tokens = tokenizer(text, padding='max_length',
                                   max_length=512, truncation='longest_first')
                input_ids = torch.Tensor(tokens['input_ids'])
                token_type_ids = torch.Tensor(tokens['token_type_ids'])
                attention_mask = torch.Tensor(tokens['attention_mask'])

                nsp_data = torch.stack(
                    [input_ids, token_type_ids, attention_mask], dim=0)
                input_train.append(nsp_data)

                if i == 20:
                    break

            label_train = torch.Tensor(label_train)
            input_train = torch.stack(input_train)

            label_test = []
            input_test = []
            for i, lin in enumerate(hyperpartisan['validation']):
                text = re.sub('<[^>]+>', '', lin['text'])
                index = text.find('&#')
                text = text.replace("..", "")
                while index > -1:
                    idx_semicolon = index + 2
                    if index + 2 < len(text) and text[index+2].isdigit():
                        while idx_semicolon < len(text) and text[idx_semicolon] != ';':
                            idx_semicolon += 1
                            if index - idx_semicolon > 6:
                                break
                        if text[index+2:idx_semicolon].isdigit():
                            unicode_decimal = int(text[index+2:idx_semicolon])
                            unicode = f'&#%d;' % unicode_decimal
                            text = text.replace(unicode,
                                                chr(unicode_decimal))
                    index = text.find('&#')
                text = text.replace("&amp;#160", " ")
                text = text.replace("&amp;amp;", "&")
                text = text.replace("&amp;gt;", ">")

                label_test.append(int(lin['hyperpartisan']))
                tokens = tokenizer(text, padding='max_length',
                                   max_length=512, truncation='longest_first')
                input_ids = torch.Tensor(tokens['input_ids'])
                token_type_ids = torch.Tensor(tokens['token_type_ids'])
                attention_mask = torch.Tensor(tokens['attention_mask'])

                nsp_data = torch.stack(
                    [input_ids, token_type_ids, attention_mask], dim=0)
                input_test.append(nsp_data)

                if i == 5:
                    break

            label_test = torch.Tensor(label_test)
            input_test = torch.stack(input_test)

            train_datasets.append(make_classification_dataset(
                TensorDataset(input_train, label_train), task_labels=task_label))
            test_datasets.append(make_classification_dataset(
                TensorDataset(input_test, label_test), task_labels=task_label))

        elif dataset == 'pubmed':
            str_to_idx = {'INHIBITOR': 0,
                          'ANTAGONIST': 1,
                          'AGONIST': 2,
                          'DOWNREGULATOR': 3,
                          'PRODUCT-OF': 4,
                          'SUBSTRATE': 5,
                          'INDIRECT-UPREGULATOR': 6,
                          'UPREGULATOR': 7,
                          'INDIRECT-DOWNREGULATOR': 8,
                          'ACTIVATOR': 9,
                          'AGONIST-ACTIVATOR': 10,
                          'AGONIST-INHIBITOR': 11,
                          'SUBSTRATE_PRODUCT-OF': 12
                          }
            label_train = []
            input_train = []
            with jsonlines.open('../data/endtask/chemprot/train.txt') as f:
                for lin in f:
                    label_train.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_train.append(nsp_data)
            with jsonlines.open('../data/endtask/chemprot/dev.txt') as f:
                for lin in f:
                    label_train.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_train.append(nsp_data)

            label_train = torch.Tensor(label_train)
            input_train = torch.stack(input_train)

            label_test = []
            input_test = []

            with jsonlines.open('../data/endtask/chemprot/test.txt') as f:
                for lin in f:
                    label_test.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_test.append(nsp_data)

            label_test = torch.Tensor(label_test)
            input_test = torch.stack(input_test)

            train_datasets.append(make_classification_dataset(
                TensorDataset(input_train, label_train), task_labels=task_label))
            test_datasets.append(make_classification_dataset(
                TensorDataset(input_test, label_test), task_labels=task_label))

        elif dataset == 's2orc':
            str_to_idx = {'Background': 0,
                          'Uses': 1,
                          'CompareOrContrast': 2,
                          'Extends': 3,
                          'Motivation': 4,
                          'Future': 5
                          }
            label_train = []
            input_train = []
            with jsonlines.open('../data/endtask/citation_intent/train.txt') as f:
                for lin in f:
                    label_train.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_train.append(nsp_data)
            with jsonlines.open('../data/endtask/citation_intent/dev.txt') as f:
                for lin in f:
                    label_train.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_train.append(nsp_data)

            label_train = torch.Tensor(label_train)
            input_train = torch.stack(input_train)

            label_test = []
            input_test = []
            with jsonlines.open('../data/endtask/citation_intent/test.txt') as f:
                for lin in f:
                    label_test.append(str_to_idx[lin['label']])

                    tokens = tokenizer(lin['text'], padding='max_length',
                                       max_length=512, truncation='longest_first')
                    input_ids = torch.Tensor(tokens['input_ids'])
                    token_type_ids = torch.Tensor(tokens['token_type_ids'])
                    attention_mask = torch.Tensor(tokens['attention_mask'])

                    nsp_data = torch.stack(
                        [input_ids, token_type_ids, attention_mask], dim=0)
                    input_test.append(nsp_data)

            label_test = torch.Tensor(label_test)
            input_test = torch.stack(input_test)

            train_datasets.append(make_classification_dataset(
                TensorDataset(input_train, label_train), task_labels=task_label))
            test_datasets.append(make_classification_dataset(
                TensorDataset(input_test, label_test), task_labels=task_label))

    return dataset_benchmark(train_datasets, test_datasets)
