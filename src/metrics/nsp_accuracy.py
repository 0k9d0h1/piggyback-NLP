from typing import List
from avalanche.evaluation import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy


class NSPAccuracyPluginMetric(GenericPluginMetric[float, Accuracy]):
    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        super().__init__(Accuracy(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.seq_relationship_score, strategy.mb_y)


class MinibatchNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(MinibatchNSPAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "NSPAcc_MB"


class EpochNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(EpochNSPAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "NSPAcc_Epoch"


class RunningEpochNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(RunningEpochNSPAccuracy, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "RunningNSPAcc_Epoch"


class ExperienceNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(ExperienceNSPAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "NSPAcc_Exp"


class StreamNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(StreamNSPAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval"
        )

    def __str__(self):
        return "NSPAcc_Stream"


class TrainedExperienceNSPAccuracy(NSPAccuracyPluginMetric):
    def __init__(self):
        super(TrainedExperienceNSPAccuracy, self).__init__(
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
            NSPAccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return "NSPAccuracy_On_Trained_Experiences"


def nspaccuracy_metrics(
    *,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,
) -> List[NSPAccuracyPluginMetric]:
    metrics: List[NSPAccuracyPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchNSPAccuracy())

    if epoch:
        metrics.append(EpochNSPAccuracy())

    if epoch_running:
        metrics.append(RunningEpochNSPAccuracy())

    if experience:
        metrics.append(ExperienceNSPAccuracy())

    if stream:
        metrics.append(StreamNSPAccuracy())

    if trained_experience:
        metrics.append(TrainedExperienceNSPAccuracy())

    return metrics
