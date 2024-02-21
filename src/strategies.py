import torch

from avalanche.training.templates import SupervisedTemplate
from typing import Any, Iterable, Sequence


class UnlabeledPiggybackStrategy(SupervisedTemplate):
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
        self.prediction_scores, self.seq_relationship_score = self.mb_output
        labels = self.original_input_ids.masked_fill(
            self.input_ids != 103, -100)
        self.loss_mask = self._criterion(
            self.prediction_scores.view(-1, self.model.bert.config.vocab_size), labels.view(-1).type(torch.LongTensor).to('cuda'))
        self.loss_nsp = self._criterion(
            self.seq_relationship_score, self.mb_y.type(torch.LongTensor).to('cuda'))
        return self.loss_mask + self.loss_nsp

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


class EndtaskPiggybackStrategy(SupervisedTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, experiences: Any | Iterable, eval_streams: Sequence[Any | Iterable] | None = None, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return super().train(experiences, eval_streams, **kwargs)

    def forward(self):
        self.input_ids = self.mb_x[:, 0, :].type(torch.IntTensor).to('cuda')
        self.token_type_ids = self.mb_x[:, 1, :].type(torch.IntTensor).to(
            'cuda')
        self.attention_mask = self.mb_x[:, 2, :].type(torch.IntTensor).to(
            'cuda')
        task_label = self.mb_task_id[0].item()
        return self.model(input_ids=self.input_ids, token_type_ids=self.token_type_ids, attention_mask=self.attention_mask, task_label=task_label)

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
