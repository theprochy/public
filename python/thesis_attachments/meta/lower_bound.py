from queue import PriorityQueue

from depq import DEPQ

from meta.classes import State, Job


class LowerBound:

    def evaluate(self, state) -> int:
        raise NotImplementedError("This method must be implemented by a subclass.")

    def get_name(self) -> str:
        raise NotImplementedError("This method must be implemented by a subclass.")

    @staticmethod
    def _relax_jobs(s: State) -> [Job]:
        # relaxation of release times and due dates
        jobs = list(s.t.values()) + list(s.f.values())
        jobs.sort(key=lambda x: (x.r, x.d))
        jobs_relaxed = []
        d_max = 0
        for job in jobs:
            if job.d < d_max:
                jobs_relaxed.append(Job(job.id, job.p, job.r, d_max))
            else:
                d_max = job.d
                jobs_relaxed.append(job)
        return jobs_relaxed


class DummyLB(LowerBound):

    def evaluate(self, s) -> int:
        return len(s.l)

    def get_name(self) -> str:
        return "DummyLB"


class MooreHodgsonLB(LowerBound):

    def evaluate(self, s) -> int:
        jobs = self._relax_jobs(s)
        n = len(jobs)
        S = DEPQ()
        S_prime = PriorityQueue()
        q = {}
        q_S = 0
        for j in range(n):
            # modify() from the paper
            r = 0 if j == 0 else jobs[j].r - jobs[j - 1].r
            while r > 0 and len(S) > 0:
                i = S.poplast()[0]
                q_S -= q[i.id]
                if q[i.id] <= r:
                    S_prime.put((q[i.id], i))
                    r -= q[i.id]
                else:
                    q[i.id] -= r
                    S.insert(i, q[i.id])
                    q_S += q[i.id]
                    r = 0
            # END modify() from the paper
            q[jobs[j].id] = jobs[j].p
            S.insert(jobs[j], q[jobs[j].id])
            q_S += q[jobs[j].id]
            if q_S > jobs[j].d - jobs[j].r:
                l = S.popfirst()[0]
                q_S -= q[l.id]

        return len(jobs) - len(S) - S_prime.qsize() + len(s.l)

    def get_name(self) -> str:
        return "MooreHodgsonLB"


class KiseIbarakiMineLB(LowerBound):

    def evaluate(self, s) -> int:
        jobs_relaxed = self._relax_jobs(s)

        # Kise, Ibaraki, Mine algorithm,
        # A Solvable Case of the One-Machine Scheduling Problem with Ready and Due Times,1978
        E = []
        F = 0
        for job in jobs_relaxed:
            # if F(E_{j-1} U {j}) <= d(j), then E_j = E_{j-1} U {j}
            if max(F, job.r) + job.p <= job.d:
                E.append(job)
                F = max(F, job.r) + job.p

            # else E_j = E_{j-1} U {j} - {l} where l satisfies condition (4) from Kise, Ibaraki, Mine (1978)
            else:
                # Subalgorithm to compute job l in condition (4)
                # Step 1
                if len(E) == 0:
                    continue
                F_S = {}
                F_S_prime = {}
                E.append(job)
                m = len(E)
                i = 2
                F_S[i] = E[1].r + E[1].p
                F_S_prime[i] = E[0].r + E[0].p
                h = 0
                for i in range(2, m + 1):
                    # Step 2
                    if F_S[i] > F_S_prime[i]:
                        F_S[i] = F_S_prime[i]
                        h = i - 1
                    if i == m:
                        break
                    # Step 3
                    F_S[i + 1] = max(F_S[i], E[i].r) + E[i].p
                    F_S_prime[i + 1] = max(F_S_prime[i], E[i - 1].r) + E[i - 1].p
                del E[h]
                F = F_S[i]

        return len(jobs_relaxed) - len(E) + len(s.l)

    def get_name(self) -> str:
        return "KiseIbarakiMineLB"
