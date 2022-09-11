from queue import PriorityQueue

from meta.classes import State, Job


class Proposition:

    def evaluate(self, state):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def get_name(self):
        raise NotImplementedError("This method must be implemented by a subclass.")


# Proposition 2 - Processed before relations, Proposition 11 - tightening of job windows
class Proposition2(Proposition):

    def evaluate(self, state: State):
        precedes = {}
        K = list(state.t.values())
        K.sort(key=lambda j: (j.r, j.d))

        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                if K[i].r < K[j].r and K[i].d < K[j].d:
                    satisfies = True
                    for k in range(len(K)):
                        if k == i or k == j:
                            continue
                        if K[k].r >= K[i].d:
                            break
                        if K[k].d <= K[j].r:
                            continue
                        satisfies = False
                        break
                    if satisfies:
                        if not K[i].id in precedes:
                            precedes[K[i].id] = []
                        precedes[K[i].id].append(K[j].id)

        for job_id, following_ids in precedes.items():
            job = state.t[job_id]
            # Elimination rule - tightening of release and due dates
            for follower_id in following_ids:
                if follower_id not in state.t.keys():
                    continue
                follower = state.t[follower_id]
                new_d_i = min(job.d, follower.d - follower.p)
                new_r_j = max(follower.r, job.r + job.p)
                if new_d_i < job.r + job.p or new_r_j < 0 or new_r_j > follower.d - follower.p:
                    continue
                new_i = Job(job.id, job.p, job.r, new_d_i)
                new_j = Job(follower.id, follower.p, new_r_j, follower.d)
                state.t[job.id] = new_i
                state.t[follower.id] = new_j

    def get_name(self):
        return "Proposition2"


class Proposition211(Proposition):

    def evaluate(self, state):
        L1 = PriorityQueue()
        L2 = PriorityQueue()
        for job in state.t.values():
            L1.put((job.p + job.r, job.id, job))
            L2.put((job.d - job.p, job.id, job))
        while not L1.empty() and not L2.empty():
            j_1 = L1.queue[0][2]
            j_2 = L2.queue[0][2]
            if j_1.id == j_2.id:
                L1.get()
                continue
            if j_1.r + j_1.p <= j_2.d - j_2.p:
                L1.get()
            else:
                L2.get()
                # Elimination rule - tightening of release and due dates
                job = j_2
                follower = j_1
                new_d_i = min(job.d, follower.d - follower.p)
                new_r_j = max(follower.r, job.r + job.p)
                if new_d_i < job.r + job.p or new_r_j < 0 or new_r_j > follower.d - follower.p:
                    continue
                new_i = Job(job.id, job.p, job.r, new_d_i)
                new_j = Job(follower.id, follower.p, new_r_j, follower.d)
                state.t[job.id] = new_i
                state.t[follower.id] = new_j

    def get_name(self):
        return "Proposition211"


# Proposition 3 - Dominance property, certain jobs have to be on time
# Proposition 4 - Certain jobs have to be late
class Propositions34(Proposition):

    def evaluate(self, state):
        K = list(state.t.values()) + list(state.f.values())
        K.sort(key=lambda _: _.p)

        for i in range(len(K)):
            for j in range(i + 1, len(K)):
                if K[j].r + K[j].p >= K[i].r + K[i].p and K[j].d - K[j].p <= K[i].d - K[i].p:
                    # Prop. 3: if j is on time => i is on time too
                    if K[j].id in state.t.keys() and K[i].id in state.f.keys():
                        state.f.pop(K[i].id)
                        state.t[K[i].id] = K[i]
                    # Prop. 4: if we cannot schedule both, the less interesting one is late
                    if K[i].id in state.f.keys() and K[j].id in state.f.keys():
                        if K[i].r + K[i].p + K[j].p > K[j].d and K[j].r + K[j].p + K[i].p > K[i].d:
                            state.f.pop(K[j].id)
                            state.l[K[j].id] = K[j]

    def get_name(self):
        return "Propositions3_4"
