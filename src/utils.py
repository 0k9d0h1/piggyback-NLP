import torch
import numpy as np

from modnets.layers import Binarizer
from avalanche.training.plugins import SupervisedPlugin
from typing import Optional, Any
from copy import deepcopy


class BestModelPlugin(SupervisedPlugin):

    def __init__(
        self,
    ):
        super().__init__()

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step: Optional[int] = None

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None
        self.best_step = None

    def before_training_epoch(self, strategy, **kwargs):
        ub = self._update_best(strategy)
        print("best:", ub, "\n")
        if ub is None or self.best_step is None:
            return

    def after_training(self, strategy: Any, *args, **kwargs):
        ub = self._update_best(strategy)
        print("best:", ub, "\n")
        strategy.model.load_state_dict(self.best_state)

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        names = [k for k in res.keys() if k.startswith(
            "Loss_Exp/eval_phase/val_stream")]
        if len(names) == 0:
            return None

        task_label = strategy.experience.task_label
        full_name = 'Loss_Exp/eval_phase/val_stream/Task%s/Exp%s' % (
            str(task_label).zfill(3), str(task_label).zfill(3))
        print("full_name:", full_name, "\n")
        val_loss = res.get(full_name)
        if self.best_val is None or val_loss < self.best_val:
            self.best_state = deepcopy(strategy.model.state_dict())
            if self.best_val is None:
                self.best_val = val_loss
                self.best_step = 0
                return None

            if val_loss < self.best_val:
                self.best_step = self._get_strategy_counter(strategy)
                self.best_val = val_loss

        return self.best_val

    def _get_strategy_counter(self, strategy):
        return strategy.clock.train_exp_epochs


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


def check(model, pretrained, train_ln):
    """Makes sure that the trained model weights match those of the pretrained model."""
    print('Making sure filter weights have not changed.')
    module_list = list(pretrained.modules())
    i = 0

    for module in model.modules():
        if i == len(module_list):
            break

        if 'ElementWiseLinear' in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            # Using small threshold of 1e-8 for any floating point inconsistencies.
            # Note that threshold per element is even smaller as the 1e-8 threshold
            # is for sum of absolute differences.
            assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                'module %s failed check' % (module)
            if module.bias is not None:
                bias = module.bias.data.cpu()
                bias_pretrained = module_list[i].bias.data.cpu()
                assert (bias - bias_pretrained).abs().sum() < 1e-8

        elif 'Embedding' in str(type(module)) and 'Bert' not in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                'module %s failed check' % (module)

        elif 'LayerNorm' in str(type(module)):
            if not train_ln:
                weight = module.weight.data.cpu()
                weight_pretrained = module_list[i].weight.data.cpu()
                # Using small threshold of 1e-8 for any floating point inconsistencies.
                # Note that threshold per element is even smaller as the 1e-8 threshold
                # is for sum of absolute differences.
                assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                    'module %s failed check' % (module)
                if module.bias is not None:
                    bias = module.bias.data.cpu()
                    bias_pretrained = module_list[i].bias.data.cpu()
                    assert (bias - bias_pretrained).abs().sum() < 1e-8

        elif 'Dict' in str(type(module)) or 'BertForPreTraining' in str(type(module)):
            continue
        i += 1
    print('Passed checks...')


def ckpt_masks(model, dat, train_ln):
    dataset2masks = {}
    dataset2ln = {}
    for data_idx, data in enumerate(dat):
        masks = {}
        ln = {}
        for module_idx, module in enumerate(model.modules()):
            if 'ElementWise' in str(type(module)):
                mask = Binarizer.apply(module.masks[str(data_idx)])
                mask = mask.data.detach().cpu()

                num_zero = mask.eq(0).sum()
                num_one = mask.eq(1).sum()
                total = mask.numel()
                print(data_idx, module_idx, num_zero / total * 100)

                assert num_zero + num_one == total
                mask = mask.type(torch.ByteTensor)
                masks[module_idx] = np.packbits(mask.numpy(), axis=0)
            elif 'LayerNorm' in str(type(module)):
                if train_ln:
                    ln[module_idx] = (module.weight,)
                    if module.bias is not None:
                        ln[module_idx] += module.bias

        dataset2masks[data] = masks
        dataset2ln[data] = ln

    return dataset2masks, dataset2ln
