from typing import List
from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metrics import LossMetric
from avalanche.evaluation import GenericPluginMetric


class MLLossPluginMetric(GenericPluginMetric[float, LossMetric]):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = LossMetric()
        super(MLLossPluginMetric, self).__init__(
            self._loss, reset_at, emit_at, mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._loss.update(strategy.loss_mask, patterns=len(strategy.mb_y))


class MinibatchMLLoss(MLLossPluginMetric):
    def __init__(self):
        super(MinibatchMLLoss, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "MLLoss_MB"


class EpochMLLoss(MLLossPluginMetric):
    def __init__(self):
        super(EpochMLLoss, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "MLLoss_Epoch"


class RunningEpochMLLoss(MLLossPluginMetric):
    def __init__(self):
        super(RunningEpochMLLoss, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RunningMLLoss_Epoch"


class ExperienceMLLoss(MLLossPluginMetric):
    def __init__(self):
        super(ExperienceMLLoss, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "MLLoss_Exp"


class StreamMLLoss(MLLossPluginMetric):
    def __init__(self):
        super(StreamMLLoss, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "MLLoss_Stream"


def mlloss_metrics(
    *, minibatch=False, epoch=False, epoch_running=False, experience=False, stream=False
) -> List[MLLossPluginMetric]:

    metrics: List[MLLossPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchMLLoss())

    if epoch:
        metrics.append(EpochMLLoss())

    if epoch_running:
        metrics.append(RunningEpochMLLoss())

    if experience:
        metrics.append(ExperienceMLLoss())

    if stream:
        metrics.append(StreamMLLoss())

    return metrics
