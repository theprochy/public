#!/usr/bin/env python3

import sys
import time
import numpy as np
from math import factorial
from itertools import permutations, chain, combinations

from Template import Node, DirEdge, Graph, Agent


class Game:
    def banzhaf(self):
        start = time.time()
        self.G.get_max_flow()
        t_max_flow = time.time() - start
        banzhaf = np.zeros(self.n)
        if self.n < 10 and (2 ** self.n) * self.n * t_max_flow < 1:
            for i in range(self.n):
                a = self.agents[i]
                N_a = self.agents[:]
                N_a.remove(a)
                for A in chain.from_iterable(combinations(N_a, r) for r in range(self.n+1)):
                    E_cur = set()
                    for agent in A:
                        E_cur |= self.A[agent]
                    v_A = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                    E_cur |= self.A[a]
                    v_A_i = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                    banzhaf[i] = banzhaf[i] + v_A_i - v_A
            return banzhaf / (2 ** (self.n - 1))
        else:
            start = time.time()
            m = 0
            while time.time() < start + 4:
                m = m + 1
                A = np.random.randint(2, size=self.n)
                E_cur = set()
                for i in range(self.n):
                    if A[i]:
                        E_cur |= self.A[self.agents[i]]
                v_A = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                for i in range(self.n):
                    E_cur = set()
                    if A[i]:
                        for j in range(self.n):
                            if A[j] and i != j:
                                E_cur |= self.A[self.agents[j]]
                        v_A_i = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                        banzhaf[i] = banzhaf[i] + v_A - v_A_i
                    else:
                        for j in range(self.n):
                            if A[j] or i == j:
                                E_cur |= self.A[self.agents[j]]
                        v_A_i = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                        banzhaf[i] = banzhaf[i] + v_A_i - v_A
            return banzhaf / m

    def shapley(self):
        start = time.time()
        self.G.get_max_flow()
        t_max_flow = time.time() - start
        shapley = np.zeros(self.n)
        if self.n < 7 and factorial(self.n) * self.n * t_max_flow < 9:
            for perm in permutations(range(self.n)):
                E_cur = set()
                v_A = 0
                for i in range(self.n):
                    E_cur |= self.A[Agent(perm[i] + 1)]
                    v_A_i = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                    shapley[perm[i]] = shapley[perm[i]] + v_A_i - v_A
                    v_A = v_A_i
            return shapley / factorial(self.n)
        else:
            start = time.time()
            m = 0
            while time.time() < start + 15:
                m = m + 1
                perm = np.random.permutation(np.arange(self.n))
                E_cur = set()
                v_A = 0
                for i in range(self.n):
                    E_cur |= self.A[Agent(perm[i] + 1)]
                    v_A_i = self.G.get_induced_graph_by_edges(E_cur).get_max_flow()
                    shapley[perm[i]] = shapley[perm[i]] + v_A_i - v_A
                    v_A = v_A_i
            return shapley / m


class FlowGame(Graph, Game):
    def __init__(self, A=None, G=None):
        super().__init__(A=A, G=G)
        self.n = max([a.id for a in A.keys()])
        self.A = A
        self.G = G
        self.agents = [Agent(i + 1) for i in range(self.n)]


def prepare(io):
    size_of_V, size_of_E, size_of_N = map(int, io.readline().split(" "))
    G = Graph()
    V = {id: Node(id) for id in range(size_of_V)}
    A = {Agent(id): set() for id in range(1, size_of_N + 1)}
    for i in range(size_of_E):
        tail_id, head_id, cap_e, agent_id = map(int, io.readline().split(" "))
        a = Agent(agent_id)
        u, v = V[tail_id], V[head_id]
        e = DirEdge(tail=u, head=v, l=0, u=cap_e)
        G.add_edge(e)
        A[a] |= {e}
    return A, G


def output(shapley, banzhaf):
    print(*shapley)
    print(*banzhaf)


def run(FG):
    return FG.shapley(), FG.banzhaf()


if __name__ == "__main__":
    A, G = prepare(sys.stdin)
    FG = FlowGame(A, G)
    shapley, banzhaf = run(FG)
    output(shapley, banzhaf)
