import math
import random

import torch

from meta.classes import Job


class Chooser:
                                                # True - tardy, False - early
    def choose(self, state, jobs, probs) -> (Job, bool):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def get_name(self):
        raise NotImplementedError("This method must be implemented by a subclass.")


class RandomChooser(Chooser):
    def choose(self, state, jobs, probs) -> (Job, bool):
        return random.choice(list(state.f.values())), bool(random.randint(0, 1))

    def get_name(self):
        return "RandomChooser"


class GreedyChooser(Chooser):

    def choose(self, state, jobs, probs) -> Job:
        return list(jobs.values())[torch.argmax(probs).item()], True

    def get_name(self):
        return "GreedyChooser"


class OptimisticGreedyAbsValueChooser(Chooser):

    def choose(self, state, jobs, probs) -> (Job, bool):
        keys = list(jobs.keys())
        if torch.min(probs[keys]) > 0.5:
            argmax = torch.argmax(probs[keys])
            return jobs[keys[argmax]], True
        argmin = torch.argmin(probs[keys])
        return jobs[keys[argmin]], False


    def get_name(self):
        return "OptimisticGreedyAbsValueChooser"


class GreedyAbsValueChooser(Chooser):

    def choose(self, state, jobs, probs) -> (Job, bool):
        keys = list(jobs.keys())
        argmax = torch.argmax(torch.abs(probs[keys] - 0.5))
        return jobs[keys[argmax]], bool(round(probs[keys[argmax]].item()))

    def get_name(self):
        return "GreedyAbsValueChooser"


class BaptistePeridyPinsonChooser(Chooser):

    def choose(self, state, jobs, probs) -> (Job, bool):
        p_min = math.inf
        for j in state.f.values():
            if j.p < p_min:
                p_min = j.p
        window = 0
        chosen = None
        for j in state.f.values():
            if j.p >= 1.1 * p_min:
                continue
            if j.d - j.r > window:
                window = j.d - j.r
                chosen = j
        return chosen, False

    def get_name(self):
        return "BaptistePeridyPinsonChooser"


class EnhancedBaptisteChooser(Chooser):

    def choose(self, state, jobs, probs) -> (Job, bool):
        keys = torch.tensor(list(jobs.keys()))
        cur_probs = probs[keys]
        on_time = keys[cur_probs < 0.5]
        late = keys[cur_probs >= 0.5]
        if torch.numel(on_time) > 0:
            return self._choose(on_time, jobs)
        return self._choose(late, jobs)

    @staticmethod
    def _choose(keys, jobs):
        p_min = math.inf
        for k in keys:
            j = jobs[k.item()]
            if j.p < p_min:
                p_min = j.p
        window = 0
        chosen = None
        for k in keys:
            j = jobs[k.item()]
            if j.p >= 1.1 * p_min:
                continue
            if j.d - j.r > window:
                window = j.d - j.r
                chosen = j
        return chosen, False

    def get_name(self):
        return "EnhancedBaptisteChooser"
