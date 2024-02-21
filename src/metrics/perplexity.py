import torch
import numpy as np

from typing import List
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean


class Perplexity(Metric[float]):
    def __init__(self):
        self._mean_accuracy = Mean()

    @torch.no_grad()
    def update(
        self,
        perplexity,
        mb_y
    ) -> None:
        self._mean_accuracy.update(perplexity, len(mb_y))

    def result(self) -> float:
        return self._mean_accuracy.result()

    def reset(self) -> None:
        self._mean_accuracy.reset()


class PerplexityPluginMetric(GenericPluginMetric[float, Perplexity]):
    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        super().__init__(Perplexity(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(np.exp(strategy.loss_mask.item()), strategy.mb_y)


class MinibatchPerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(MinibatchPerplexity, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Perplexity_MB"


class EpochPerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(EpochPerplexity, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "Perplexity_Epoch"


class RunningEpochPerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(RunningEpochPerplexity, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RunningPerplexity_Epoch"


class ExperiencePerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(ExperiencePerplexity, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "Perplexity_Exp"


class StreamPerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(StreamPerplexity, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "Perplexity_Stream"


class TrainedExperiencePerplexity(PerplexityPluginMetric):
    def __init__(self):
        super(TrainedExperiencePerplexity, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )
        self._current_experience = 0

    def after_training_exp(self, strategy):
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        self.reset()
        return super().after_training_exp(strategy)

    def update(self, strategy):
        if strategy.experience.current_experience <= self._current_experience:
            PerplexityPluginMetric.update(self, strategy)

    def __str__(self):
        return "Perplexity_On_Trained_Experiences"


def perplexity_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[PerplexityPluginMetric]:

    metrics: List[PerplexityPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchPerplexity())

    if epoch:
        metrics.append(EpochPerplexity())

    if epoch_running:
        metrics.append(RunningEpochPerplexity())

    if experience:
        metrics.append(ExperiencePerplexity())

    if stream:
        metrics.append(StreamPerplexity())

    if trained_experience:
        metrics.append(TrainedExperiencePerplexity())

    return metrics
