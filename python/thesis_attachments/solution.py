"""

"""
import json
import random
import warnings
from typing import Tuple, Dict, Sequence, Union

import numpy as np
from bson import ObjectId

from result import CompleteResult, Result
from ulits.common_utils import warn_with_traceback, first, second, tupleit, third, sublist

Number = Union[float, int]

warnings.showwarning = warn_with_traceback


class Instance:

    def __init__(self, td: Tuple[Tuple[Number, Number, Number]]):
        """

        :rtype: Instance
        """
        self._not_nan_instance = None
        self._p = None
        self._r = None
        self._d = None
        self.td = td
        self.td: Tuple[Tuple[Number, Number, Number]]

    def __len__(self):
        return len(self.td)

    @property
    def proc(self) -> Tuple[Number]:
        if not self._p:
            self._p = tuple(map(first, self.td))
            self._p: Tuple[Number]
        return self._p

    @property
    def release(self) -> Tuple[Number]:
        if not self._r:
            self._r = tuple(map(second, self.td))
            self._r: Tuple[Number]
        return self._r

    @property
    def due(self) -> Tuple[Number]:
        if not self._d:
            self._d = tuple(map(third, self.td))
            self._d: Tuple[Number]
        return self._d

    @property
    def n(self):
        return len(self.td)

    @staticmethod
    def from_lists(proc: Sequence[Number], release: Sequence[Number], due: Sequence[Number]):
        return Instance(tuple(zip(proc, release, due)))

    def __getitem__(self, item):
        return self.td[item]

    def __eq__(self, other):
        return self.td == other.td

    def __hash__(self):
        return self.td.__hash__()

    def to_mongo(self):
        return self.td

    def to_json(self):
        """
        """
        return self.td

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str([self.proc, self.release, self.due])


DUMMY_INSTANCE = Instance.from_lists([1., 2., 3., 4.],
                                     [1, 2, 1, 2],
                                     [3, 4, 5, 6],
                                     )

OPTIMAL_ESTIMATORS = ['SlackILPEst()', 'SlackILPEst(due)', 'SlackILPEst(-due)', 'SlackILPEst(proc)',
                      'SlackILPEst(proc)',
                      'PremekILPSecEst(sc())', 'PremekILPSecEst(sc(proc))', 'PremekILPSecEst(sc(-proc))',
                      'PremekILPSecEst(sc(due))', 'PremekILPSecEst(sc(-due))', 'DecompositionEst()']

FORCE_METRICS = []


class Solution:
    """
    Class implementing one instance of single machine total tardiness problem. With field dictionary results
    to save result of different method to heuristic (or optimal) solution.
    """

    def __init__(self, instance=None, _id=None, train=False, lock=False, alpha=None, beta=None, p_max=None,
                 results=None, descriptors=None, valid=True, sets=None, full_results=None, generator=None, mmax=None,
                 load=None, **kwargs):
        """
        :param instance: Instance
        :param results: dictionary of results for each Estimator
        :param _id: mongodb id
        :param rdd: rdd from json
        :param tf: tf from json
        :param train: flag marking test / train dataset
        :param lock: flag use for locking instance between process
        """
        self.mmax = int(mmax)
        self.load = float(load)
        self.alpha = alpha
        self.beta = beta
        self.p_max = p_max
        self.generator = generator
        if descriptors is None:
            descriptors = {}
        self.descriptors = descriptors
        self.valid = valid
        if not results:
            results = {}
        self.results = results  # type: Dict[str, CompleteResult]
        # Deserialization from mongo
        self.instance, self.results, self.full_results, _id = self.deserialize_from_mongo(instance, results,
                                                                                          full_results, _id)
        self._id = _id
        # End of deserialization

        self.train = train
        self.lock = lock
        self.sets = sets

    @staticmethod
    def deserialize_from_mongo(instance, results, full_results, _id):
        if type(_id) is str:
            _id = ObjectId(_id)
        if type(instance) is list or type(instance) is tuple:
            instance = Instance(tupleit(instance))

        out_res = {}
        for k, v in results.items():
            if type(v) is dict:
                out_res[k] = CompleteResult(**v)
            # if type(v) is list: # for old backup
            #     out_res[k] = CompleteResult(*v)
            else:
                out_res[k] = v
        return instance, out_res, full_results, _id

    @staticmethod
    def min_init(**kwargs):
        """
        For create of Instance require only processing times and due dates.

        :param kwargs:
        :return: Instance with due dates and processing times
        """
        if sublist(['proc', 'release', 'due'], kwargs):
            kwargs['instance'] = Instance.from_lists(kwargs['proc'], kwargs['release'], kwargs['due'])
        if 'td' in kwargs:
            kwargs['instance'] = Instance(kwargs['td'])

        return Solution(**kwargs)

    @property
    def n(self):
        """
        :return:         return length of instance

        """
        return len(self.instance)

    @property
    def test(self):
        return not self.train

    def __str__(self, old=False):
        """

        :return:
        """
        out = ["n: {}".format(self.n)]
        out += [f'{str(k)}: {str(v)}' for k, v in self.__dict__.items()]

        return '{Solution: ' + ', '.join(out) + '}'

    def to_mongo(self):
        if not self.sets:
            self.sets = set()
        sets = set(self.sets)
        d = {
            'n': self.n,
            'descriptors': self.descriptors
        }

        for k in list(self.results.keys()):
            if '.' in k:
                self.results[k.replace('.', '_')] = self.results.pop(k)

        # from targets.estimators.estimator_config import estimator_factory
        # ei = estimator_factory()
        # ref_est = ei.get_group('referent')
        # d['error'] = {}
        # for re in ref_est:
        #     if re.referent and re.name() in self.results:
        #         ref_est = self.results[re.name().replace('.', '_')].criterion
        #         d['error'][re.name()] = {k: abs(v.criterion - ref_est) for k, v in self.results.items()}

        opt = self.optimal_result()
        d['has_optimal_solution'] = opt is not None
        metrics = {}
        from targets.estimators.estimator_config import estimator_factory
        est_factory = estimator_factory()
        if opt:
            if not self.check_optimal_results_same(opt):
                sets.add('OPTIMAL_ERROR')
            opt_c = opt.criterion

            if opt_c is not None:
                for est_n, est_res in self.results.items():
                    if not est_res.empty() or est_n.startswith("NNDP"):
                        self.evaluate_metrics(est_n, est_res, metrics, FORCE_METRICS)
                        if est_res.metrics:
                            metrics_arr = est_res.metrics
                            self.evaluate_metrics(est_n, est_res, metrics, metrics_arr, est_n.startswith("NNDP") or est_n.startswith("Dec") or est_n.startswith("EarlyTardy"))
                        else:
                            if est_n not in est_factory:
                                # print('====================== skip ================')
                                # print(est_n)
                                # print('====================== skip ================')
                               continue
                            est = est_factory[est_n]
                            metrics_arr = est.metrics
                            self.evaluate_metrics(est_n, est_res, metrics, metrics_arr)
        d['metrics'] = metrics

        sets_def = {}

        for k, v in sets_def.items():
            if v(self):
                sets.update(k)
        d.update(self.__dict__)
        # if any_in([''], sets):
        #     sets.add('test_easy')
        d['sets'] = list(sets)
        d['instance'] = self.instance.to_mongo()
        results = {}
        for k, v in self.results.items():
            results[k.replace('.', '_')] = v.to_mongo()
        d['results'] = results
        if not d['_id']:
            d.pop('_id')
        return d

    def evaluate_metrics(self, est_n, est_res, metrics, metrics_arr, bonus_into_metrics=False):
        from targets.metrics_config import METRICS
        if bonus_into_metrics:
            if est_res.bonus is not None:
                for m in metrics_arr:
                    if m not in metrics:
                        metrics[m] = {}
                    metrics[m][est_n] = METRICS[m].calculate(self, est_res.bonus, self.n)
                est_res.bonus = None
        else:
            for m in metrics_arr:
                if m not in metrics:
                    metrics[m] = {}
                metrics[m][est_n] = METRICS[m].calculate(self.optimal_result().criterion, est_res.criterion, self.n)

    def check_optimal_results_same(self, opt):
        opt_c = opt.criterion
        for est in OPTIMAL_ESTIMATORS:
            if est in self.results:
                if opt_c != self.results[est].criterion:
                    return True
        return False

    def has_optimal_result(self):
        return True if self.optimal_result() else False

    def optimal_result(self):
        for est in self.results.keys():
            if est in OPTIMAL_ESTIMATORS or "Decomposition" in est:
                return self.results[est]

        return None

    def result_to_et(self, result):
        return [0] * self.n

    def to_json(self):
        """
        Return instance dict, enable json serialization of instance

        :return: self.__dict__
        """
        return self.__dict__

    def sort_no(self, arr=None):
        """
        :param arr:
        :return:
        """
        return self, arr

    def self_copy(self):
        """
        :return:
        """
        return Solution(**json.loads(json.dumps(self)))

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return self.self_copy()

    def to_ndarray(self) -> np.ndarray:
        """
        Return instance as nd array [proc, due]
        """
        return np.array(self.instance.td)

    def get_tardy_jobs(self, opt: Result):
        t = 0
        tardy = []
        for j in opt.order:
            t = max(t, self.instance.release[j])
            t += self.instance.proc[j]
            if t > self.instance.due[j]:
                tardy.append(True)
            else:
                tardy.append(False)
        if sum(tardy) != opt.criterion:
            raise ValueError('opt crit non equal')
        return tardy


def generate_sample_valente(pmax, n, alpha, beta) -> Solution:
    inst = generate_instance_valente(n, pmax, alpha, beta)
    return Solution(inst, alpha=alpha, beta=beta, p_max=pmax, generator='valente')


def generate_instance_valente(n, p_max, alpha, beta) -> Instance:
    # generating processing times from [0;p_max]
    p_i = [random.randint(1, p_max + 1) for _ in range(n)]

    r_i = [random.randint(1, round(beta * sum(p_i))) for _ in range(n)]
    s_i = [random.randint(1, round(alpha * sum(p_i))) for _ in range(n)]

    d_i = tuple(np.array(r_i) + np.array(p_i) + np.array(s_i))

    return Instance.from_lists(p_i, r_i, d_i)


def generate_sample_baptiste(pmax, n, mmax, load):
    inst = generate_by_sampling_min_sum_starting_minus_release_baptiste(pmax, n, mmax, load)
    return Solution(inst, mmax=mmax, load=load)


def generate_by_sampling_min_sum_starting_minus_release_baptiste(p_max, size, m_max, load, p_min=0):
    p_i = np.array([random.randint(max(p_min, 1), p_max + 1) for _ in range(size)]).astype(int)
    s_i = np.array([random.randint(0, m_max + 1) for _ in range(size)]).astype(int)
    # sigma = np.abs((size * (p_max)) / (load * 8) - p_max / 4 - m_max / 4)
    sigma = max(0, (size * (p_min + p_max) / (2 * load) - p_max - m_max) / 4)
    r_i = np.random.normal(0, sigma, size)
    r_i = np.round(r_i).astype(int)
    r_i[r_i < 0] = 0
    d_i = tuple(np.array(r_i) + np.array(p_i) + np.array(s_i))
    return Instance.from_lists(list(map(int, p_i)), list(map(int, r_i)), list(map(int, d_i)))
