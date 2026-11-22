"""2-апта: BFS, UCS және A* 8x8 торда.

Ұяшыққа өту құны ластануға байланысты:
    c(v) = (1 - alpha) + alpha * pm(v) / pm_ref
alpha = 0 болса - қадам санына тең (ең қысқа жол), alpha = 1 болса - таза ауа басымдығы.
BFS құнды ескермейді (ең аз қадам), UCS пен A* құн бойынша оңтайлы.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config as C

Cell = tuple[int, int]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class SearchResult:
    algorithm: str
    path: list[Cell]
    cost: float                 # жолдың құны (бірдей құн картасымен өлшенген)
    steps: int                  # жол ұзындығы (қадам саны)
    expanded: int               # ашылған түйіндер саны
    expanded_order: list[Cell] = field(default_factory=list)
    time_ms: float = 0.0

    @property
    def found(self) -> bool:
        return bool(self.path)


def step_cost_map(pm: np.ndarray, alpha: float) -> np.ndarray:
    """Әр ұяшыққа кірудің құны."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 0 мен 1 аралығында болуы керек")
    pm_ref = float(np.mean(pm))
    return (1.0 - alpha) + alpha * pm / pm_ref


def validate(layout: np.ndarray, start: Cell, goal: Cell) -> None:
    n = layout.shape[0]
    for name, (r, c) in (("Бастау", start), ("Аяқтау", goal)):
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"{name} ұяшығы тордан тыс: {(r, c)}")
        if layout[r, c] == C.BUILDING:
            raise ValueError(f"{name} ұяшығы тосқауыл (ғимарат): {(r, c)}")


def neighbors(layout: np.ndarray, cell: Cell, blocked: set | None = None):
    n = layout.shape[0]
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and layout[nr, nc] != C.BUILDING \
                and not (blocked and (nr, nc) in blocked):
            yield (nr, nc)


def path_cost(path: list[Cell], cost_map: np.ndarray) -> float:
    return float(sum(cost_map[r, c] for r, c in path[1:]))


def _reconstruct(parent: dict, goal: Cell) -> list[Cell]:
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def _finish(name, parent, goal, cost_map, order, t0) -> SearchResult:
    ms = (time.perf_counter() - t0) * 1000
    if goal not in parent:
        return SearchResult(name, [], float("inf"), 0, len(order), order, ms)
    path = _reconstruct(parent, goal)
    return SearchResult(name, path, path_cost(path, cost_map), len(path) - 1, len(order), order, ms)


def bfs(layout, cost_map, start: Cell, goal: Cell, blocked=None) -> SearchResult:
    validate(layout, start, goal)
    t0 = time.perf_counter()
    parent = {start: None}
    queue, order = deque([start]), []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        if cur == goal:
            break
        for nb in neighbors(layout, cur, blocked):
            if nb not in parent:
                parent[nb] = cur
                queue.append(nb)
    return _finish("BFS", parent, goal, cost_map, order, t0)


def _best_first(name, layout, cost_map, start, goal, h, blocked=None) -> SearchResult:
    validate(layout, start, goal)
    t0 = time.perf_counter()
    g = {start: 0.0}
    parent = {start: None}
    counter = 0
    heap = [(h(start), counter, start)]
    closed, order = set(), []
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        order.append(cur)
        if cur == goal:
            break
        for nb in neighbors(layout, cur, blocked):
            ng = g[cur] + float(cost_map[nb])
            if nb not in g or ng < g[nb] - 1e-12:
                g[nb] = ng
                parent[nb] = cur
                counter += 1
                heapq.heappush(heap, (ng + h(nb), counter, nb))
    return _finish(name, parent, goal, cost_map, order, t0)


def ucs(layout, cost_map, start: Cell, goal: Cell, blocked=None) -> SearchResult:
    return _best_first("UCS", layout, cost_map, start, goal, lambda n: 0.0, blocked)


def heuristic_factory(layout, cost_map, goal: Cell, blocked=None):
    """h(n) = Manhattan(n, goal) * c_min. c_min - өтуге болатын ұяшықтардағы ең кіші қадам құны,
    сондықтан h допустимді және консистентті (A* оңтайлы)."""
    passable = layout != C.BUILDING
    if blocked:
        passable = passable.copy()
        for r, c in blocked:
            passable[r, c] = False
    c_min = float(cost_map[passable].min())
    gr, gc = goal
    return lambda n: (abs(n[0] - gr) + abs(n[1] - gc)) * c_min


def astar(layout, cost_map, start: Cell, goal: Cell, blocked=None) -> SearchResult:
    validate(layout, start, goal)
    h = heuristic_factory(layout, cost_map, goal, blocked)
    return _best_first("A*", layout, cost_map, start, goal, h, blocked)


ALGORITHMS = {"BFS": bfs, "UCS": ucs, "A*": astar}


def run(algorithm: str, layout, pm, alpha, start, goal, blocked=None) -> SearchResult:
    """Қолайлы кіру нүктесі: pm картасынан құнды есептеп, таңдалған алгоритмді іске қосады."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Белгісіз алгоритм: {algorithm}")
    if blocked and (start in blocked or goal in blocked):
        raise ValueError("Бастау немесе аяқтау ұяшығы тосқауыл етілген")
    return ALGORITHMS[algorithm](layout, step_cost_map(pm, alpha), start, goal, blocked)
