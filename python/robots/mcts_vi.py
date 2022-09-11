#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import itertools
import numpy as np
import gurobipy as g
from pathlib import Path

import pickle

import GridMap as gmap

PURSUER = 1
EVADER = 2

GREEDY = "GREEDY"
MONTE_CARLO = "MONTE_CARLO"
VALUE_ITERATION = "VALUE_ITERATION"

env = g.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

def compute_nash(matrix):
    """
    Method to calculate the value-iteration policy action

    Parameters
    ----------
    matrix: n times m array of floats
        Game utility matrix


    Returns
    -------
    value:float
        computed value of the game
    strategy:float[n]
        probability of player 1 playing each action in nash equilibrium
    """
    n_a1, n_a2 = matrix.shape
    m = g.Model(env=env)
    U = m.addVar(lb=-math.inf, ub=math.inf, obj=1.0, vtype=g.GRB.CONTINUOUS, name="U")
    x = m.addVars(n_a1, vtype=g.GRB.CONTINUOUS, name="x")
    m.addConstr(g.quicksum(x.values()) == 1)
    for i in range(n_a2):
        m.addConstr(g.LinExpr(matrix[:, i], x.values()) >= U)
    m.setObjective(g.LinExpr(1, U), g.GRB.MAXIMIZE)
    m.optimize()
    return m.objVal, [x.x for x in x.values()]


class Player:
    def __init__(self, robots, role, policy=GREEDY, color='r', epsilon=1,
                 timeout=5.0, game_name=None):
        """ constructor of the Player class
        Args: robots: list((in,int)) - coordinates of individual player's robots
              role: PURSUER/EVADER - player's role in the game
              policy: GREEDY/MONTE_CARLO/VALUE_ITERATION - player's policy,
              color: string - player color for visualization
              epsilon: float - [0,1] epsilon value for greedy policy
              timeout: float - timout for MCTS policy
              game_name: string - name of the currently played game
        """
        # list of the player's robots
        self.robots = robots[:]
        # next position of the player's robots
        self.next_robots = robots[:]

        if role == "EVADER":
            self.role = EVADER
        elif role == "PURSUER":
            self.role = PURSUER
        else:
            raise ValueError('Unknown player role')

        # selection of the policy
        if policy == GREEDY:
            self.policy = self.greedy_policy
        elif policy == MONTE_CARLO:
            self.policy = self.monte_carlo_policy
            self.timeout = timeout * len(self.robots)  # MCTS planning timeout
            self.tree = {}
            self.max_depth = 10
            self.step = 0
            self.max_steps = 100
            self.beta = 0.95
            self.c = 1
        elif policy == VALUE_ITERATION:
            self.policy = self.value_iteration_policy
            # values for the value iteration policy
            self.loaded_policy = None
            self.gamma = 0.95
        else:
            raise ValueError('Unknown policy')

        # parameters
        self.color = color  # color for plotting purposes
        self.game_name = game_name  # game name for loading vi policies

    #####################################################
    # Game interface functions
    #####################################################
    def add_robot(self, pos):
        """ method to add a robot to the player
        Args: pos: (int,int) - position of the robot
        """
        self.robots.append(pos)
        self.next_robots.append(pos)

    def del_robot(self, pos):
        """ method to remove the player's robot
        Args: pos: (int,int) - position of the robot to be removed
        """
        idx = self.robots.index(pos)
        self.robots.pop(idx)
        self.next_robots.pop(idx)

    def calculate_step(self, gridmap, evaders, pursuers):
        """ method to calculate the player's next step using selected policy
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in
                       the game (except the player's robots, if he is pursuer)
        """
        self.policy(gridmap, evaders, pursuers)

    def take_step(self):
        """ method to perform the step
        """
        self.robots = self.next_robots[:]

    #####################################################
    #####################################################
    # GREEDY POLICY
    #####################################################
    #####################################################
    def greedy_policy(self, gridmap, evaders, pursuers, epsilon=1):
        """ Method to calculate the greedy policy action
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in
                       the game (except the player's robots, if he is pursuer)
              epsilon: float (optional) - optional epsilon-greedy parameter
        """
        self.next_robots = self.robots[:]

        rnd = random.uniform(0, 1)
        rndm = True if rnd < 1 - epsilon else False
        # for each of player's robots plan their actions
        for idx in range(0, len(self.robots)):
            robot = self.robots[idx]
            neighbors = gridmap.neighbors4(robot)
            random.shuffle(neighbors)
            if rndm:
                self.next_robots[idx] = neighbors[0]
                continue

            if self.role == PURSUER:
                closest = np.inf
                for evader in evaders:
                    if gridmap.dist(robot, evader) < closest:
                        closest = gridmap.dist(robot, evader)
                        closest_pos = evader
                closest = np.inf
                for next in neighbors:
                    if gridmap.dist(closest_pos, next) < closest:
                        closest = gridmap.dist(closest_pos, next)
                        next_pos = next
                self.next_robots[idx] = next_pos

            if self.role == EVADER:
                closest = np.inf
                closest_pos = []
                for pursuer in pursuers:
                    d = gridmap.dist(robot, pursuer)
                    if d < closest:
                        closest = d
                        closest_pos = [pursuer]
                    elif d == closest:
                        closest_pos.append(pursuer)
                farthest = -1
                for next in neighbors:
                    min_d = np.inf
                    for pos in closest_pos:
                        if gridmap.dist(next, pos) < min_d:
                            min_d = gridmap.dist(next, pos)
                    if min_d > farthest:
                        farthest = min_d
                        farthest_pos = next
                self.next_robots[idx] = farthest_pos

    #####################################################
    #####################################################
    # MONTE CARLO TREE SEARCH POLICY
    #####################################################
    #####################################################
    class MCNode:
        def __init__(self, pursuers, evaders, terminal):
            self.w = 0
            self.n = 0
            self.terminal = terminal
            self.pursuers = pursuers
            self.evaders = evaders
            self.children = {}

    def simulate(self, gridmap, pursuers, evaders):
        eps = .85
        players = []
        if self.role == PURSUER:
            players.append(Player(pursuers, "PURSUER", policy=GREEDY, epsilon=eps))
            players.append(Player(evaders, "EVADER", policy=GREEDY, epsilon=eps))
        else:
            players.append(Player(pursuers, "PURSUER", policy=GREEDY, epsilon=eps))
            players.append(Player(evaders, "EVADER", policy=GREEDY, epsilon=eps))
        pursuit_game = Game.Game(gridmap, players)
        n_steps = 10

        for i in range(0, n_steps):
            if not pursuit_game.is_end():
                pursuit_game.step()

        if not pursuit_game.is_end():
            # evader win
            if self.role == EVADER:
                return True
            return False
        else:
            # pursuer win
            if self.role == PURSUER:
                return True
            return False

    def monte_carlo_policy(self, gridmap, evaders, pursuers):
        """ Method to calculate the monte carlo tree search policy action
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in
                       the game (except the player's robots, if he is pursuer)
        """
        catches = set()
        non_catches = set()

        def mcts(node, depth, depth_limit):
            if depth == depth_limit:
                ret = self.simulate(gridmap, node.pursuers, node.evaders)
                node.n = node.n + 1
                node.w = node.w + int(ret)
                return ret
            if len(node.children) == 0:
                new_pursuers, new_evaders = [], []
                for pursuer in node.pursuers:
                    new_pursuers.append(gridmap.neighbors4(pursuer))
                for evader in node.evaders:
                    new_evaders.append(gridmap.neighbors4(evader))
                pursuers_t = tuple(node.pursuers)
                evaders_t = tuple(node.evaders)

                for next_pos in itertools.product(itertools.product(*new_pursuers), list(itertools.product(*new_evaders))):
                    if ((pursuers_t, evaders_t), next_pos) in catches:
                        node.children[next_pos] = self.MCNode([], [], True)
                        node.n = node.n + 1
                        if self.role == PURSUER:
                            node.w = node.w + 1
                    elif ((pursuers_t, evaders_t), next_pos) in non_catches:
                        node.children[next_pos] = self.MCNode(list(next_pos[0]), list(next_pos[1]), False)
                    else:
                        caught = False
                        if self.role == EVADER:
                            for i in range(len(node.evaders)):
                                if caught:
                                    break
                                evader = next_pos[1][i]
                                evader_neighbors = gridmap.neighbors4(evader)
                                if evader in node.pursuers:
                                    catches.add(((pursuers_t, evaders_t), next_pos))
                                    node.children[next_pos] = self.MCNode([], [], True)
                                    node.n = node.n + 1
                                    caught = True
                                    break
                                for j in range(len(node.pursuers)):
                                    if node.pursuers[j] in evader_neighbors:
                                        catches.add(((pursuers_t, evaders_t), next_pos))
                                        node.children[next_pos] = self.MCNode([], [], True)
                                        node.n = node.n + 1
                                        caught = True
                                        break
                        else:
                            for i in range(len(node.pursuers)):
                                if caught:
                                    break
                                for j in range(len(node.evaders)):
                                    if next_pos[0][i] == next_pos[1][j] or (node.pursuers[i] == next_pos[1][j] and node.evaders[j] == next_pos[0][i]):
                                        catches.add(((pursuers_t, evaders_t), next_pos))
                                        node.children[next_pos] = self.MCNode([], [], True)
                                        node.n = node.n + 1
                                        node.w = node.w + 1
                                        caught = True
                                        break
                        if not caught:
                            non_catches.add(((pursuers_t, evaders_t), next_pos))
                            node.children[next_pos] = self.MCNode(list(next_pos[0]), list(next_pos[1]), False)
            if np.all([n.terminal for n in node.children.values()]):
                node.terminal = True
                node.n = node.n + 1
                if self.role == PURSUER:
                    node.w = node.w + 1
                    return True
                return False
            for child in node.children.values():
                if child.terminal:
                    continue
                if child.n == 0:
                    ret = self.simulate(gridmap, child.pursuers, child.evaders)
                    child.n = child.n + 1
                    child.w = child.w + int(ret)
                    node.n = node.n + 1
                    node.w = node.w + int(ret)
                    return ret
            best_next, best_ucts = None, 0
            for child in node.children.values():
                if not child.terminal:
                    ucts = child.w / child.n + c * math.sqrt(math.log(node.n) / child.n)
                    if ucts > best_ucts:
                        best_ucts = ucts
                        best_next = child
            ret = mcts(best_next, depth + 1, depth_limit)
            node.n = node.n + 1
            node.w = node.w + int(ret)
            return ret

        self.next_robots = self.robots[:]

        # measure the time for selecting next action
        clk = time.time()
        c, depth_limit = 3, 4
        if self.role == EVADER:
            evaders = copy.deepcopy(self.robots)
        else:
            pursuers = copy.deepcopy(self.robots)
        root = self.MCNode(pursuers, evaders, False)
        res = self.simulate(gridmap, pursuers, evaders)
        root.n = root.n + 1
        root.w = root.w + int(res)

        while (time.time() - clk) < self.timeout*len(self.robots):
        #for i in range(1000):
            mcts(root, 0, depth_limit)
        best_acc = 0
        next_robots = None
        for node in root.children.values():
            if not node.terminal and node.w / node.n > best_acc:
                best_acc = node.w / node.n
                if self.role == EVADER:
                    next_robots = node.evaders
                else:
                    next_robots = node.pursuers
        self.next_robots = next_robots

    #####################################################
    #####################################################
    # VALUE ITERATION POLICY
    #####################################################
    #####################################################
    def init_values(self, gridmap):
        mapping_i2c = {}
        mapping_c2i = {}
        count = 0
        for i in range(gridmap.width):
            for j in range(gridmap.height):
                if gridmap.passable((i, j)):
                    mapping_i2c[count] = (i, j)
                    mapping_c2i[(i, j)] = count
                    count += 1
        return mapping_i2c, mapping_c2i, count

    def random_policy(self, coord_state, gridmap, mapping_c2i, role):
        a, b, c = coord_state
        neigh_a = gridmap.neighbors4(a)
        neigh_b = gridmap.neighbors4(b)
        neigh_c = gridmap.neighbors4(c)
        if role == PURSUER:
            combined_actions = []
            for action_one in neigh_b:
                for action_two in neigh_c:
                    combined_actions.append((mapping_c2i[action_one], mapping_c2i[action_two]))
            return combined_actions, [1 / len(combined_actions)] * len(combined_actions)
        else:
            combined_actions = []
            for action in neigh_a:
                combined_actions.append(mapping_c2i[action])
            return combined_actions, [1 / len(combined_actions)] * len(combined_actions)

    def compute_vi_policy(self, c2i, i2c, spaces, gridmap):
        # player 1 max player pursuer
        pursuer_policy, evader_policy = {}, {}
        v_old = np.zeros((spaces, spaces, spaces))
        v = np.ones((spaces, spaces, spaces))
        while True:
            for a in range(spaces):
                for b in range(spaces):
                    for c in range(spaces):
                        s = (i2c[a], i2c[b], i2c[c])
                        e_n, p1_n, p2_n = gridmap.neighbors4(s[0]), gridmap.neighbors4(s[1]), gridmap.neighbors4(s[2])
                        Q = np.zeros((len(p1_n) * len(p2_n), len(e_n)))
                        for i in range(len(e_n)):
                            for j in range(len(p1_n)):
                                for k in range(len(p2_n)):
                                    r = 1 if (e_n[i] == p1_n[j] or e_n[i] == p2_n[k]) or\
                                                (s[0] == p1_n[j] and e_n[i] == s[1]) or (s[0] == p2_n[k] and e_n[i] == s[2])\
                                            else 0
                                    if r == 1:
                                        Q[j * len(p2_n) + k, i] = 1
                                    else:
                                        Q[j * len(p2_n) + k, i] = 0.95 * v_old[c2i[e_n[i]], c2i[p1_n[j]], c2i[p2_n[k]]]
                        v[a, b, c], _ = compute_nash(Q)
            #print(np.max(np.abs(v - v_old)))
            if np.max(np.abs(v - v_old)) < 0.0001:
                break
            v_old = v
            v = np.zeros((spaces, spaces, spaces))
        for a in range(spaces):
            for b in range(spaces):
                for c in range(spaces):
                    s = (i2c[a], i2c[b], i2c[c])
                    e_n, p1_n, p2_n = gridmap.neighbors4(s[0]), gridmap.neighbors4(
                        s[1]), gridmap.neighbors4(s[2])
                    Q = np.zeros((len(p1_n) * len(p2_n), len(e_n)))
                    for i in range(len(e_n)):
                        for j in range(len(p1_n)):
                            for k in range(len(p2_n)):
                                r = 1 if (e_n[i] == p1_n[j] or e_n[i] == p2_n[k] or
                                          (s[0] == p1_n[j] and e_n[i] == s[1]) or (
                                                      s[0] == p2_n[k] and e_n[i] == s[2])) \
                                    else 0
                                Q[j * len(p2_n) + k, i] = r + self.gamma * v_old[
                                    c2i[e_n[i]], c2i[p1_n[j]], c2i[p2_n[k]]]
                    _, prob = compute_nash(Q)
                    pursuer_policy[a, b, c] = ([(c2i[p], c2i[q]) for p in p1_n for q in p2_n], prob)
                    _, prob = compute_nash(-Q.T)
                    evader_policy[a, b, c] = ([c2i[p] for p in e_n], prob)

        return v, evader_policy, pursuer_policy

    def value_iteration_policy(self, gridmap, evaders, pursuers):
        """ Method to calculate the value-iteration policy action
        Args: gridmap: GridMap - map of the environment
              evaders: list((int,int)) - list of coordinates of evaders in the
                            game (except the player's robots, if he is evader)
              pursuers: list((int,int)) - list of coordinates of pursuers in
                       the game (except the player's robots, if he is pursuer)
        """
        self.next_robots = self.robots[:]

        # if there are not precalculated values for policy
        if not self.loaded_policy:
            policy_file = Path(os.path.join(os.getcwd(), "policies", self.game_name) + ".policy")
            # if there is policy file, load it...
            if policy_file.is_file():
                # load the strategy file
                self.loaded_policy = pickle.load(open(str(policy_file), 'rb'))
            # ...else calculate the policy
            else:
                mapping_i2c, mapping_c2i, count = self.init_values(gridmap)
                values, evader_policy, pursuer_policy = \
                    self.compute_vi_policy(mapping_c2i, mapping_i2c, count, gridmap)
                self.loaded_policy = (values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i)
                pickle.dump(self.loaded_policy, open(str(policy_file), 'wb'))

        values, evader_policy, pursuer_policy, mapping_i2c, mapping_c2i = self.loaded_policy

        if self.role == PURSUER:
            state = (mapping_c2i[evaders[0]], mapping_c2i[self.robots[0]], mapping_c2i[self.robots[1]])
        else:
            state = (mapping_c2i[self.robots[0]], mapping_c2i[pursuers[0]], mapping_c2i[pursuers[1]])

        if self.role == PURSUER:
            action_index = np.random.choice(tuple(range(len(pursuer_policy[state][0]))), p=pursuer_policy[state][1])
            action = pursuer_policy[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action[0]]
            self.next_robots[1] = mapping_i2c[action[1]]
        else:
            action_index = np.random.choice(tuple(range(len(evader_policy[state][0]))), p=evader_policy[state][1])
            action = evader_policy[state][0][action_index]
            self.next_robots[0] = mapping_i2c[action]
            #####################################################
