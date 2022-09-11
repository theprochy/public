import math
import random
import time
from queue import PriorityQueue

import torch

from typing import List

from result import Result, CompleteResult
from solution import Instance
from meta.oracle import MLOracleStatic
from meta.classes import State, Job
from estimators.estimator import Estimator
from ml.dp.metrics.dpmetrics import ETMetrics


class EarlyTardyEstimator(Estimator):

    def __init__(self, oracle: MLOracleStatic, rerun_nn_steps: int = -1, early_threshold: float = 0.5, lds: int = 0):
        self.oracle = oracle
        self.n_removed = 0
        self.rerun_nn_steps = rerun_nn_steps
        self.early_threshold = early_threshold
        self.best = [], []
        self.lds = lds
        self.lds_cache = {}
        super().__init__()

    @staticmethod
    def _instance_to_list(instance: Instance) -> [Job]:
        return [Job(i, instance.proc[i], instance.release[i], instance.due[i]) for i in range(instance.n)]

    def _estimate(self, instance: Instance, opt_result: CompleteResult = None) -> Result:
        return self._estimate_with_timeout(instance)

    @Estimator.exit_after(3600)
    def _estimate_with_timeout(self, instance: Instance) -> Result:
        self.n_removed = 0
        start = time.monotonic()
        self.oracle.calculate_probs(instance)
        nn_time = time.monotonic() - start
        start = time.monotonic()
        n_start = torch.sum(self.oracle.probs < self.early_threshold).item()
        self._solve_instance(instance)
        T_final, sigma_final = self.best
        alg_time = time.monotonic() - start
        order = [job.id for job in sigma_final]
        return Result(order, instance.n - len(T_final), T_final,
                      ETMetrics(instance.n, len(T_final), n_start, nn_time,
                                alg_time / (self.n_removed + 1)))

    def debug(self, instance: Instance) -> (int, int, int, [int], [Job], float):
        start = time.time()
        self.oracle.calculate_probs(instance)
        print('############################################################')
        self._solve_instance(instance)
        T_final, sigma_final = self.best
        print('n:', instance.n)
        print('n_start:', torch.sum(self.oracle.probs < self.early_threshold).item())
        print('Sigma:', set(j.id for j in sigma_final))
        print(time.time() - start)
        print(instance.n - len(T_final), 'tardy jobs')
        return instance.n - len(T_final), T_final, sigma_final, time.time() - start

    def pre_sort(self, instance: Instance) -> Instance:
        return instance

    def _get_info(self) -> str:
        return ""

    @property
    def metrics(self) -> List[str]:
        return ['ETStartRelError()', 'ETRelError()', 'ETGapPct()', 'ETNStart()', 'ETNNTime()',
                'ETTimePerRepetition()', 'ETEarlyPct()']

    def name(self) -> str:
        return "EarlyTardyEstimatorRerun" + str(self.rerun_nn_steps) + "Th" + str(self.early_threshold) + "Lds" + str(self.lds)

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return "EarlyTardyEstimatorRerun" + str(args[0]) + "Th" + str(args[1]) + "Lds" + str(args[2])

    @staticmethod
    def _schrage(I: [Job]) -> ([int], [Job]):
        # Schrage algorithm (EDD sequence)
        I.sort(key=lambda j: j.r)
        n = len(I)
        t = 0
        idx = 0
        T = []
        sigma = []
        ready = PriorityQueue()
        for i in range(n):
            if ready.empty() and idx < n and I[idx].r > t:
                t = I[idx].r
            while idx < n and I[idx].r <= t:
                ready.put((I[idx].d, I[idx].p, I[idx]))
                idx += 1
            cur = ready.get()[2]
            T.append(t)
            sigma.append(cur)
            t += cur.p
        return T, sigma

    @staticmethod
    def _feasible_edd(T: [int], sigma: [Job]) -> bool:
        assert len(T) == len(sigma)
        for i in range(len(T)):
            assert T[i] >= sigma[i].r
            if T[i] + sigma[i].p > sigma[i].d:
                return False
        return True

    @staticmethod
    def _carlier_identify(T: [int], sigma: [Job]) -> (bool, Job, Job, [Job]):
        # Identify whether the schedule is optimal or there exists a job c and set J
        assert len(T) == len(sigma)
        n = len(T)
        L_max = 0
        L_max_idx = None
        for i in range(n):
            cur = sigma[i]
            if T[i] + cur.p - cur.d > L_max:
                L_max = T[i] + cur.p - cur.d
                L_max_idx = i
        r_min = math.inf
        d_max = 0
        sum_p = 0
        J = []
        for i in range(L_max_idx, -1, -1):
            J.append(sigma[i])
            sum_p += sigma[i].p
            if sigma[i].r < r_min:
                r_min = sigma[i].r
            if sigma[i].d > d_max:
                d_max = sigma[i].d
            # One of the following ifs should be reached according to Carlier
            # We reach the end of a chain
            if T[i - 1] + sigma[i - 1].p < T[i]:
                # assert optimality
                if r_min + sum_p - d_max != L_max:
                    assert False
                return True, None, None, []
            # The previous job is the critical one c
            if r_min + sum_p - d_max + sigma[i - 1].p > L_max:
                c = sigma[i - 1]
                return False, c, sigma[L_max_idx], J
        assert False

    def _carlier_append(self, q: [State], state: State, T: [int], sigma: [Job]):
        optimal, c, sigma_i, J = self._carlier_identify(T, sigma)
        if optimal:
            return
        # schedule c after J
        r_min = math.inf
        sum_p = 0
        for j in J:
            sum_p += j.p
            if j.r < r_min:
                r_min = j.r
        r_new = r_min + sum_p
        if r_new > c.r and r_new + c.p <= c.d:
            c_new = Job(c.id, c.p, r_new, c.d)
            t_new = state.t.copy()
            t_new[c.id] = c_new
            q.append(State(t_new, state.l.copy(), state.f.copy()))

        # schedule c before J
        d_new = sigma_i.d - sum_p
        if c.r + c.p <= d_new < c.d:
            c_new = Job(c.id, c.p, c.r, d_new)
            t_new = state.t.copy()
            t_new[c.id] = c_new
            q.append(State(t_new, state.l, state.f))

    def _is_feasible(self, state: State):
        T_final, sigma_final = None, None
        T, sigma = self._schrage(list(state.t.values()))
        if self._feasible_edd(T, sigma):
            return T, sigma
        q_carlier = []
        self._carlier_append(q_carlier, state, T, sigma)
        cont = True
        while len(q_carlier) > 0 and cont:
            state_carlier = q_carlier.pop()
            T_carlier, sigma_carlier = self._schrage(list(state_carlier.t.values()))
            if self._feasible_edd(T_carlier, sigma_carlier):
                T_final, sigma_final = T_carlier, sigma_carlier
                cont = False
            else:
                self._carlier_append(q_carlier, state_carlier, T_carlier, sigma_carlier)
        return T_final, sigma_final

    def _solve_instance(self, instance: Instance):
        jobs = [Job(i, instance.proc[i], instance.release[i], instance.due[i]) for i in range(instance.n)]
        cur_idx = [i for i in range(instance.n) if self.oracle.probs[i].item() < self.early_threshold]
        if not cur_idx:
            return [], [], 0
        return self._solve_instance_recursive(jobs, cur_idx, 0, self.lds)

    def _solve_instance_recursive(self, jobs, cur_idx, discrepancy, max_discrepancy):
        if len(cur_idx) < len(self.best[0]):
            return
        cur_idx.sort()
        cur_jobs = [jobs[idx] for idx in cur_idx]
        state = State({j.id: j for j in cur_jobs}, {}, {})
        if tuple(cur_idx) in self.lds_cache:
            return
        else:
            T_final, sigma_final = self._is_feasible(state)
            self.lds_cache[tuple(cur_idx)] = T_final, sigma_final
            if T_final:
                if len(T_final) > len(self.best[0]):
                    self.best = T_final, sigma_final
                return
        if len(cur_idx) == 1:
            return [], []
        order = cur_idx.copy()
        order.sort(key=lambda idx: self.oracle.probs[idx])
        for cur_disc in range(0, max_discrepancy - discrepancy + 1, 1):
            # if iteration > 0 and self.rerun_nn_steps != -1 and iteration % self.rerun_nn_steps == 0:
            #     new_instance = Instance(tuple([(jobs[idx].p, jobs[idx].r, jobs[idx].d) for idx in cur_idx]))
            #     self.oracle.calculate_probs(new_instance)
            #     cur_idx = list(range(len(cur_idx)))
            if 1 + cur_disc > len(cur_idx):
                continue
            new_cur_idx = order.copy()
            del new_cur_idx[-1 - cur_disc]
            self._solve_instance_recursive(jobs, new_cur_idx, discrepancy + cur_disc, max_discrepancy)


if __name__ == "__main__":
    # inst = Instance(((1, 0, 2),(1, 0, 1), (1, 4, 6), (4, 0, 8)))
    # inst = Instance(((11, 0, 23), (10, 0, 28), (5, 0, 7), (2, 0, 14), (9, 0, 25), (11, 0, 29)))
    # inst = Instance(((2, 0, 8), (3, 0, 8), (2, 0, 28), (4, 5, 14), (3, 1, 6), (3, 0, 14), (3, 0, 27), (6, 4, 31), (4, 0, 16), (3, 3, 28), (11, 0, 24), (1, 0, 27), (4, 14, 21), (3, 11, 16), (1, 9, 11)))
    # inst = Instance(((2, 0, 8), (3, 0, 8), (2, 0, 28), (4, 5, 14), (3, 1, 6), (3, 0, 14), (3, 0, 27), (6, 4, 31), (4, 0, 16), (3, 3, 28), (11, 0, 24), (1, 0, 27), (4, 14, 21), (3, 11, 16), (1, 9, 11), (11, 0, 23), (10, 0, 28), (5, 0, 7), (2, 0, 14), (9, 0, 25), (11, 0, 29)))
    # inst = Instance(((3, 0, 11), (10, 0, 12), (3, 8, 18), (4, 4, 8), (10, 0, 11), (10, 16, 30), (2, 0, 10), (11, 0, 16), (2, 7, 15), (8, 0, 21), (5, 5, 18), (4, 0, 19), (6, 0, 16), (7, 4, 16)))
    # inst = Instance([[10, 6, 21], [10, 0, 27], [5, 8, 33], [1, 0, 18], [9, 0, 17], [4, 13, 43], [7, 24, 50], [4, 0, 28], [3, 0, 3], [6, 0, 11], [5, 1, 6], [10, 0, 30], [2, 10, 18], [5, 7, 15], [10, 4, 38], [10, 0, 30], [8, 0, 18], [8, 0, 17], [1, 39, 40], [7, 0, 30], [3, 21, 38], [1, 0, 5], [5, 11, 23], [7, 15, 39], [10, 0, 11], [7, 1, 29], [9, 0, 26], [6, 14, 42], [11, 33, 69], [4, 0, 12]])
    # inst = Instance(
    # [[7, 50, 93], [6, 8, 31], [5, 40, 81], [9, 0, 20], [5, 55, 93], [10, 0, 16], [9, 0, 29], [6, 0, 26], [3, 0, 3],
    #  [6, 10, 46], [4, 0, 9], [9, 30, 75], [11, 24, 51], [8, 0, 14], [2, 9, 25], [10, 0, 35], [10, 24, 36],
    #  [2, 9, 26], [11, 0, 52], [8, 0, 46], [11, 0, 52], [2, 0, 26], [2, 51, 72], [11, 12, 41], [5, 0, 34],
    #  [8, 101, 145], [6, 0, 25], [3, 61, 70], [4, 122, 140], [1, 0, 4], [4, 29, 34], [1, 23, 50], [3, 11, 26],
    #  [10, 34, 63], [9, 0, 30], [9, 2, 33], [7, 15, 55], [11, 5, 37], [11, 0, 50], [8, 11, 35], [2, 0, 26],
    #  [3, 13, 16], [1, 0, 3], [6, 0, 18], [2, 0, 5], [4, 0, 32], [6, 0, 43], [1, 61, 102], [8, 28, 59],
    #  [8, 70, 118]])
    inst = Instance(((11, 0, 13), (9, 3, 15), (8, 12, 23), (8, 14, 26), (11, 0, 17), (8, 10, 18), (3, 53, 60), (3, 0, 7), (4, 0, 4), (6, 57, 66), (6, 0, 7), (3, 9, 17), (1, 21, 28), (6, 0, 8), (11, 0, 13), (2, 15, 21), (4, 0, 8), (4, 0, 6), (5, 14, 23), (10, 6, 21), (2, 5, 11), (2, 12, 16), (4, 0, 6), (10, 0, 11), (1, 36, 42), (1, 0, 4), (2, 0, 5), (5, 6, 15), (11, 3, 18), (2, 0, 8), (9, 0, 15), (7, 0, 12), (3, 0, 9), (1, 8, 13), (4, 13, 22), (6, 0, 12), (2, 0, 3), (3, 0, 3), (1, 0, 5), (9, 0, 11)))


    # inst = Instance(((11, 0, 24), (3, 0, 19), (2, 0, 15), (9, 0, 32), (1, 0, 4), (8, 0, 25), (5, 0, 17), (9, 2, 24), (11, 0, 27), (8, 0, 23), (9, 0, 30), (7, 0, 25), (2, 2, 6), (4, 1, 30)))
    #inst = Instance(((4, 0, 11), (11, 0, 11), (11, 6, 24), (7, 0, 17), (6, 6, 23), (5, 16, 21)))

    oracle = MLOracleStatic()
    #est1 = EarlyTardyEstimator(oracle, -1, 0.4)
    #est1.debug(inst)

    est2 = EarlyTardyEstimator(oracle, -1, 0.5, 2)
    est2.debug(inst)
