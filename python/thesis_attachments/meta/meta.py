import copy
import math
from queue import PriorityQueue

from meta.classes import Job, State
from meta.oracle import MLOracleStatic

from solution import Instance


class Meta:

    def __init__(self, oracle, chooser, propositions, lower_bounds, carlier_full, later_lb=None):
        self.oracle = oracle
        self.chooser = chooser
        self.propositions = propositions
        self.lower_bounds = lower_bounds
        self.carlier_full = carlier_full
        self.later_lb = later_lb
        self.nodes = 0
        self.nodes_uncut = 0

    def select_job(self, state, jobs):
        probs = self.oracle.tardy_probability(state, jobs)
        return self.chooser.choose(state, jobs, probs)

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
                # break ties arbitrarily or based on processing time?
                ready.put((I[idx].d, I[idx]))
                idx += 1
            cur = ready.get()[1]
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

    def solve_instance(self, instance: Instance):
        self.nodes = 0
        self.nodes_uncut = 0
        jobs = [Job(i, instance.proc[i], instance.release[i], instance.due[i]) for i in range(instance.n)]
        ret = self.solve(jobs)
        return ret


    """
        Main method which performs the B&B

        :param I - set of all jobs

        """

    def solve(self, I: [Job]):
        n_jobs = len(I)
        init = State({}, {}, {j.id: j for j in I})
        q = [init]
        best = math.inf
        best_solution = ([], [])

        while len(q) != 0:
            self.nodes += 1
            state = q.pop()

            #
            # Computation of lower bound(s)
            #
            cont = False
            for lower_bound in self.lower_bounds:
                if best < math.inf and lower_bound.evaluate(state) >= best:
                    cont = True
                    break
            if cont:
                continue
            self.nodes_uncut += 1

            #
            # Application of dominance properties
            #

            # Proposition 1 - Problem decomposition
            if len(state.f) > 5:
                I_s = []
                K = [(j.r, True, j) for j in state.t.values()] + [(j.d, False, j) for j in state.t.values()] \
                    + [(j.r, True, j) for j in state.f.values()] + [(j.d, False, j) for j in state.f.values()]
                K.sort(key=lambda x: (x[0], x[1], x[2].id))
                d = 0
                I_p = []
                for time in K:
                    if time[1]:
                        d = d + 1
                        I_p.append(time[2])
                    else:
                        d = d - 1
                        if d == 0:
                            I_s.append(I_p)
                            I_p = []

                if len(I_s) > 1:
                    T = []
                    sigma = []
                    for I_p in I_s:
                        T_p, sigma_p = self.solve(I_p)
                        T += T_p
                        sigma += sigma_p

                    assert len(T) == len(sigma)
                    criterion = len(state.l) + len(state.f) + len(state.t) - len(T)
                    if criterion < best:
                        best = criterion
                        best_solution = (T, sigma)
                    continue

            # Evaluation of optional propositions

            for proposition in self.propositions:
                proposition.evaluate(state)

            if self.later_lb:
                cont = False
                for lower_bound in self.later_lb:
                    if best < math.inf and lower_bound.evaluate(state) >= best:
                        cont = True
                        break
                if cont:
                    continue

            # Checking whether EDD on on-time jobs is feasible, if not Carlier branching

            T, sigma = self._schrage(list(state.t.values()))
            if self._feasible_edd(T, sigma):
                # edd feasible - chose a free job u and schedule it on time, on backtracking schedule it late
                if len(state.f) == 0:
                    # we have a solution
                    if len(state.l) < best:
                        best = len(state.l)
                        best_solution = (T, sigma)
                else:
                    chosen, tardy = self.select_job(state, state.f)
                    state.f.pop(chosen.id)
                    if tardy:
                        t_new = state.t.copy()
                        t_new[chosen.id] = chosen
                        q.append(State(t_new, state.l, state.f.copy()))
                        l_new = state.l.copy()
                        l_new[chosen.id] = chosen
                        q.append(State(state.t, l_new, state.f))
                    else:
                        l_new = state.l.copy()
                        l_new[chosen.id] = chosen
                        q.append(State(state.t, l_new, state.f.copy()))
                        t_new = state.t.copy()
                        t_new[chosen.id] = chosen
                        q.append(State(t_new, state.l, state.f))
            else:
                # try to sequence on time jobs based on Carlier
                if self.carlier_full:
                    q_carlier = []
                    self._carlier_append(q_carlier, state, T, sigma)
                    while len(q_carlier) > 0:
                        state_carlier = q_carlier.pop()
                        T_carlier, sigma_carlier = self._schrage(list(state_carlier.t.values()))
                        if self._feasible_edd(T_carlier, sigma_carlier):
                            q.append(state_carlier)
                        else:
                            self._carlier_append(q_carlier, state_carlier, T_carlier, sigma_carlier)
                else:
                    self._carlier_append(q, state, T, sigma)

        return best_solution

    def get_name(self):
        name = self.oracle.get_name()
        name += self.chooser.get_name()
        for lower_bound in self.lower_bounds:
            name += lower_bound.get_name()
        for proposition in self.propositions:
            name += proposition.get_name()
        return name
