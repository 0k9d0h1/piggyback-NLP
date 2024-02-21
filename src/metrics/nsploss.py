from typing import List
from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metrics import LossMetric
from avalanche.evaluation import GenericPluginMetric


class NSPLossPluginMetric(GenericPluginMetric[float, LossMetric]):
    def __init__(self, reset_at, emit_at, mode):
        self._loss = LossMetric()
        super(NSPLossPluginMetric, self).__init__(
            self._loss, reset_at, emit_at, mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._loss.update(strategy.loss_nsp, patterns=len(strategy.mb_y))


class MinibatchNSPLoss(NSPLossPluginMetric):
    def __init__(self):
        super(MinibatchNSPLoss, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "NSPLoss_MB"


class EpochNSPLoss(NSPLossPluginMetric):
    def __init__(self):
        super(EpochNSPLoss, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "NSPLoss_Epoch"


class RunningEpochNSPLoss(NSPLossPluginMetric):
    def __init__(self):
        super(RunningEpochNSPLoss, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RunningNSPLoss_Epoch"


class ExperienceNSPLoss(NSPLossPluginMetric):
    def __init__(self):
        super(ExperienceNSPLoss, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "NSPLoss_Exp"


class StreamNSPLoss(NSPLossPluginMetric):
    def __init__(self):
        super(StreamNSPLoss, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "NSPLoss_Stream"


def nsploss_metrics(
    *, minibatch=False, epoch=False, epoch_running=False, experience=False, stream=False
) -> List[NSPLossPluginMetric]:

    metrics: List[NSPLossPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchNSPLoss())

    if epoch:
        metrics.append(EpochNSPLoss())

    if epoch_running:
        metrics.append(RunningEpochNSPLoss())

    if experience:
        metrics.append(ExperienceNSPLoss())

    if stream:
        metrics.append(StreamNSPLoss())

    return metrics
