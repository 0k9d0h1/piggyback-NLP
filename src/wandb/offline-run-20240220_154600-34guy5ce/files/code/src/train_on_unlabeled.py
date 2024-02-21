import torch
import random
import argparse
import torch.nn as nn
import avalanche.evaluation.metrics as amet

from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from transformers import BertModel
from transformers import AutoTokenizer
from transformers import BertConfig
from modnets import BertForPreTraining
from dataset import create_unlabeled_benchmark, create_endtask_benchmark
from dataset_small import create_small_benchmark
from strategies import UnlabeledPiggybackStrategy
from metrics import mlloss_metrics, nsploss_metrics, perplexity_metrics, nspaccuracy_metrics
from utils import copy_weights, check, ckpt_masks, BestModelPlugin
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--datasets', type=str, default='realnews,s2orc,pubmed',
                   help='Names of datasets')
FLAGS.add_argument('--model', type=str, default='bert-base-uncased',
                   help='type of pretrained model')
FLAGS.add_argument('--train_ln', type=bool, default=False)
FLAGS.add_argument('--epoch', type=int, default=5)
FLAGS.add_argument('--seed', type=int, default=1)
args = FLAGS.parse_args()


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
    project_name="piggyback_NLP",
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
    mlloss_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    nsploss_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    perplexity_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    nspaccuracy_metrics(
        epoch=True,
        experience=True,
        stream=True
    ),
    loggers=[interactive_logger]
)
cl_strategy = UnlabeledPiggybackStrategy(model,
                                         optimizer,
                                         criterion,
                                         train_mb_size=16,
                                         train_epochs=args.epoch,
                                         eval_mb_size=16,
                                         device='cuda',
                                         evaluator=eval_plugin,
                                         eval_every=1,
                                         plugins=[BestModelPlugin()])

# benchmark_endtask = create_endtask_benchmark(args, tokenizer)
benchmark_unlabeled = create_small_benchmark(args, tokenizer)

# train_stream = benchmark_endtask.train_stream
# test_stream = benchmark_endtask.test_stream
train_stream = benchmark_unlabeled.train_stream
test_stream = benchmark_unlabeled.test_stream
val_stream = benchmark_unlabeled.stream_factory("val", benchmark_unlabeled)

copy_weights(model, model_pretrained)

for module in model.modules():
    if "Embedding" in str(type(module)) and "Bert" not in str(type(module)):
        for param in module.parameters():
            param.requires_grad = False
    if "LayerNorm" in str(type(module)):
        for param in module.parameters():
            param.requires_grad = args.train_ln
            print(param.requires_grad)

torch.set_printoptions(profile="full")
dat = args.datasets.split(',')

for train_exp, test_exp, val_exp in zip(train_stream, test_stream, val_stream):
    # print(model)
    print("Start of experience: ", train_exp.current_experience)
    print("Current Classes: ", train_exp.classes_in_this_experience)

    cl_strategy.train(train_exp, eval_streams=[val_exp])
    print('Training completed')

    check(model, model_pretrained, args.train_ln)

print('Computing accuracy on the test set')
result = cl_strategy.eval(test_stream)
losses = []
for i in range(len(dat)):
    losses.append(
        str(result['Loss_Exp/eval_phase/test_stream/Task%s/Exp%s' % (str(i).zfill(3), str(i).zfill(3))]))

ckpt = {
    'model': model,
    'results': result
}
torch.save(ckpt, '../checkpoint_unlabeled/BERT_piggyback_%s_%s.pt' %
           (args.datasets, ','.join(losses)))

# dataset2masks, dataset2ln = ckpt_masks(model, dat, args.train_ln)
# ckpt = {
#     'masks': dataset2masks,
#     'ln': dataset2ln,
#     'heads': model.heads,
#     'results': result
# }
# torch.save(ckpt, '../checkpoint_mask/ViT_piggyback_%s_%s.pt' %
#            (args.datasets, ','.join(losses)))
