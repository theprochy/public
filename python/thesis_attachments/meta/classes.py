from typing import NamedTuple


class Job(NamedTuple):
    id: int
    p: int
    r: int
    d: int


class State(NamedTuple):
    t: dict
    l: dict
    f: dict
