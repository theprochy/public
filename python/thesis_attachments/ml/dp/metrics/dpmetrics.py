import torch
from metrics.metric import Metric
from targets.estimators.normalizer_config import PREDICTION_NORMALIZER


class ETMetrics:

    def __init__(self, n, n_ontime, n_start, nn_time, time_per_rep):
        self.n = n
        self.n_ontime = n_ontime
        self.n_start = n_start
        self.nn_time = nn_time
        self.time_per_rep = time_per_rep


class Acc5(Metric):

    def calculate(self, truth, prediction, *args) -> float:
        truth_n = PREDICTION_NORMALIZER['Tardy()'].normalize_y(truth)
        with torch.no_grad():
            return torch.sum(torch.round(prediction) == torch.tensor(truth_n)).item() / prediction.numel()

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'Acc5()'


class Nodes(Metric):

    def calculate(self, truth, prediction, *args) -> float:
        return prediction[0]

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'Nodes()'


class NodesUncut(Metric):

    def calculate(self, truth, prediction, *args) -> int:
        return prediction[1]

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'NodesUncut()'


class ETStartRelError(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        return bonus.n_start / (bonus.n - truth.optimal_result().criterion)

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETStartRelError()'


class ETRelError(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        return bonus.n_ontime / (bonus.n - truth.optimal_result().criterion)

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETRelError()'


class ETGapPct(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        opt = truth.optimal_result().criterion
        if opt == 0:
            return 50
        return 100 * (bonus.n - bonus.n_ontime - opt) / opt

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETGapPct()'


class ETNStart(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        return bonus.n_start

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETNStart()'


class ETNNTime(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        return bonus.nn_time

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETNNTime()'


class ETTimePerRepetition(Metric):

    def calculate(self, truth, bonus, *args) -> int:
        return bonus.time_per_rep

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETTimePerRepetition()'


class ETEarlyPct(Metric):
    def calculate(self, truth, bonus, *args) -> int:
        return (bonus.n - truth.optimal_result().criterion) / bonus.n

    def name(self) -> str:
        return self.name_from_parameters()

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'ETEarlyPct()'
