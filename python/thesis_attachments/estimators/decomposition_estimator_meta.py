import random
import time
import torch

from typing import List

from result import Result, CompleteResult
from solution import Instance
from meta.meta import Meta
from meta.oracle import DummyOracle, MLOracle, MLOracleStatic, PerfectOracle
from meta.chooser import RandomChooser, GreedyChooser, GreedyAbsValueChooser, OptimisticGreedyAbsValueChooser, BaptistePeridyPinsonChooser
from meta.classes import State, Job
from meta.lower_bound import MooreHodgsonLB, DummyLB
from meta.propositions import Proposition2, Proposition211, Propositions34
from estimators.estimator import Estimator


class DecompositionEstimatorMeta(Estimator):

    def __init__(self, meta: Meta):
        super().__init__()
        self.meta = meta

    @staticmethod
    def _instance_to_list(instance: Instance) -> [Job]:
        return [Job(i, instance.proc[i], instance.release[i], instance.due[i]) for i in range(instance.n)]

    def _estimate(self, instance: Instance, opt_result: CompleteResult=None) -> Result:
        if isinstance(self.meta.oracle, MLOracleStatic):
            self.meta.oracle.calculate_probs(instance)
        if isinstance(self.meta.oracle, PerfectOracle):
            probs = torch.zeros(instance.n)
            t = 0
            for i in range(instance.n):
                j = opt_result.order[i]
                t = max(t, instance.release[j])
                t += instance.proc[j]
                if instance.due[j] < t:
                    probs[j] = 1
            self.meta.oracle.probs = probs
        return self._estimate_with_timeout(instance)

    @Estimator.exit_after(3600)
    def _estimate_with_timeout(self, instance: Instance) -> Result:
        T_final, sigma_final = self.meta.solve_instance(instance)
        order = [job.id for job in sigma_final]
        return Result(order, instance.n - len(T_final), T_final, [self.meta.nodes, self.meta.nodes_uncut])

    def debug(self, instance: Instance) -> (int, int, int, [int], [Job], float):
        start = time.time()
        T_final, sigma_final = self.meta.solve_instance(instance)
        print('############################################################')
        print(self.meta.get_name())
        print('Nodes: ', self.meta.nodes)
        print('Nodes uncut: ', self.meta.nodes_uncut)
        if isinstance(self.meta.oracle, MLOracleStatic):
            print('probs: ', torch.round(self.meta.oracle.probs))
        print('Sigma:', set(j.id for j in sigma_final))
        print(time.time() - start)
        print(instance.n - len(T_final), 'tardy jobs')
        return instance.n - len(T_final), self.meta.nodes, self.meta.nodes_uncut, T_final, sigma_final, time.time() - start

    def pre_sort(self, instance: Instance) -> Instance:
        return instance

    def _get_info(self) -> str:
        return ""

    @property
    def metrics(self) -> List[str]:
        return ['Nodes()', 'NodesUncut()']

    def name(self) -> str:
        return self.name_from_parameters(self.meta)

    @classmethod
    def name_from_parameters(cls, *args) -> str:
        return 'DecompositionEstMeta' + args[0].get_name() + "new" + '()'


if __name__ == "__main__":

    # inst = Instance(((1, 0, 2),(1, 0, 1), (1, 4, 6), (4, 0, 8)))
    # inst = Instance(((11, 0, 23), (10, 0, 28), (5, 0, 7), (2, 0, 14), (9, 0, 25), (11, 0, 29)))
    # inst = Instance(((2, 0, 8), (3, 0, 8), (2, 0, 28), (4, 5, 14), (3, 1, 6), (3, 0, 14), (3, 0, 27), (6, 4, 31), (4, 0, 16), (3, 3, 28), (11, 0, 24), (1, 0, 27), (4, 14, 21), (3, 11, 16), (1, 9, 11)))
    # inst = Instance(((2, 0, 8), (3, 0, 8), (2, 0, 28), (4, 5, 14), (3, 1, 6), (3, 0, 14), (3, 0, 27), (6, 4, 31), (4, 0, 16), (3, 3, 28), (11, 0, 24), (1, 0, 27), (4, 14, 21), (3, 11, 16), (1, 9, 11), (11, 0, 23), (10, 0, 28), (5, 0, 7), (2, 0, 14), (9, 0, 25), (11, 0, 29)))
    #inst = Instance(((3, 0, 11), (10, 0, 12), (3, 8, 18), (4, 4, 8), (10, 0, 11), (10, 16, 30), (2, 0, 10), (11, 0, 16), (2, 7, 15), (8, 0, 21), (5, 5, 18), (4, 0, 19), (6, 0, 16), (7, 4, 16)))
    inst = Instance([[10, 6, 21], [10, 0, 27], [5, 8, 33], [1, 0, 18], [9, 0, 17], [4, 13, 43], [7, 24, 50], [4, 0, 28], [3, 0, 3], [6, 0, 11], [5, 1, 6], [10, 0, 30], [2, 10, 18], [5, 7, 15], [10, 4, 38], [10, 0, 30], [8, 0, 18], [8, 0, 17], [1, 39, 40], [7, 0, 30], [3, 21, 38], [1, 0, 5], [5, 11, 23], [7, 15, 39], [10, 0, 11], [7, 1, 29], [9, 0, 26], [6, 14, 42], [11, 33, 69], [4, 0, 12]])
    #inst = Instance([[7, 50, 93], [6, 8, 31], [5, 40, 81], [9, 0, 20], [5, 55, 93], [10, 0, 16], [9, 0, 29], [6, 0, 26], [3, 0, 3], [6, 10, 46], [4, 0, 9], [9, 30, 75], [11, 24, 51], [8, 0, 14], [2, 9, 25], [10, 0, 35], [10, 24, 36], [2, 9, 26], [11, 0, 52], [8, 0, 46], [11, 0, 52], [2, 0, 26], [2, 51, 72], [11, 12, 41], [5, 0, 34], [8, 101, 145], [6, 0, 25], [3, 61, 70], [4, 122, 140], [1, 0, 4], [4, 29, 34], [1, 23, 50], [3, 11, 26], [10, 34, 63], [9, 0, 30], [9, 2, 33], [7, 15, 55], [11, 5, 37], [11, 0, 50], [8, 11, 35], [2, 0, 26], [3, 13, 16], [1, 0, 3], [6, 0, 18], [2, 0, 5], [4, 0, 32], [6, 0, 43], [1, 61, 102], [8, 28, 59], [8, 70, 118]])
    ora = MLOracleStatic()
    prop2 = Proposition2()
    prop211 = Proposition211()
    prop34 = Propositions34()
    moore_hodgson = MooreHodgsonLB()
    mt1 = Meta(DummyOracle(), BaptistePeridyPinsonChooser(), [prop2, prop34], [moore_hodgson], True)
    mt2 = Meta(PerfectOracle(50, {0, 2, 4, 9, 10, 11, 14, 17, 21, 22, 25, 27, 28, 29, 30, 31, 32, 36, 40, 41, 42, 44, 45, 46, 47, 49}),
               OptimisticGreedyAbsValueChooser(), [prop2, prop34], [moore_hodgson], True)

    test_specific = True
    if test_specific:
        print('inst.n: ', inst.n)
        DecompositionEstimatorMeta(mt1).debug(inst)
        #mt2.oracle.calculate_probs(inst)
        DecompositionEstimatorMeta(mt2).debug(inst)

    test_specific_repeatedly = False
    if test_specific_repeatedly:
        results = {i: 0 for i in range(15, 25)}
        ts, sigmas = {}, {}
        for i in range(100):
            c, n, n_c, t, sigma, tt = DecompositionEstimatorMeta(mt2).debug(inst)
            results[c] = results[c] + 1
            t = tuple(t)
            sigma = tuple(sigma)
            if sigma not in sigmas:
                sigmas[sigma] = 0
            sigmas[sigma] = sigmas[sigma] + 1
            if t not in ts:
                ts[t] = 0
            ts[t] = ts[t] + 1
        for k, v in results.items():
            print(k, v, sep=":")
        for k, v in sigmas.items():
            print(k, v, sep=":")
        for k, v in ts.items():
            print(k, v, sep=":")

    test_random = False
    if test_random:
        nodes_1, nodes_2 = {},{}
        nodes_u_1, nodes_u_2 = {}, {}
        t_1, t_2 = {}, {}
        for i in range(100, 155, 5):
            for j in range(100):
                td = []
                for k in range(i):
                    r = random.randint(0, 10 * i)
                    p = random.randint(1, 6)
                    d = r + p + random.randint(0, 6)
                    td.append((p, r, d))
                inst = Instance(tuple(td))
                mt2.oracle.calculate_probs(inst)
                print('n:', inst.n)
                c_nonmod, nodes_nonmod, uncut_nonmod, _, _, runt_1 = DecompositionEstimatorMeta(mt1).debug(inst)
                c_mod, nodes_mod, uncut_mod, _, _, runt_2 = DecompositionEstimatorMeta(mt2).debug(inst)
                if inst.n not in nodes_1:
                    nodes_1[inst.n], nodes_2[inst.n] = 0, 0
                    nodes_u_1[inst.n], nodes_u_2[inst.n] = 0, 0
                    t_1[inst.n], t_2[inst.n] = 0, 0
                nodes_1[inst.n] = nodes_1[inst.n] + nodes_nonmod
                nodes_2[inst.n] = nodes_2[inst.n] + nodes_mod
                nodes_u_1[inst.n] = nodes_u_1[inst.n] + uncut_nonmod
                nodes_u_2[inst.n] = nodes_u_2[inst.n] + uncut_mod
                t_1[inst.n] = t_1[inst.n] + runt_1
                t_2[inst.n] = t_2[inst.n] + runt_2

                if c_nonmod != c_mod:
                    print(inst.td)
                    raise Exception("Criterions do not match")
                if nodes_nonmod != nodes_mod or uncut_nonmod != uncut_mod:
                    print(inst.td)
                    print("Number of nodes does not match")
                print()
                print('############################################################')
                print('############################################################')
                print('############################################################')

        print("nodes")
        for ((k1, v1), (k2, v2)) in zip(nodes_1.items(), nodes_2.items()):
            print(str(k1) + ":" + str(v1 / 100) + " vs " + str(v2 / 100))
        print("nodes_u")
        for ((k1, v1), (k2, v2)) in zip(nodes_u_1.items(), nodes_u_2.items()):
            print(str(k1) + ":" + str(v1 / 100) + " vs " + str(v2 / 100))
        print("ts")
        for ((k1, v1), (k2, v2)) in zip(t_1.items(), t_2.items()):
            print(str(k1) + ":" + str(v1 / 100) + " vs " + str(v2 / 100))

