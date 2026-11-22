import numpy as np
import pytest

import config as C
from src import search


def random_world(seed):
    rng = np.random.default_rng(seed)
    layout = np.zeros((8, 8), dtype=int)
    layout[rng.random((8, 8)) < 0.2] = C.BUILDING
    layout[0, 0] = layout[7, 7] = C.ROAD
    pm = rng.uniform(10, 120, (8, 8))
    return layout, pm


def test_astar_cost_equals_ucs_on_random_grids():
    """A* оңтайлы: 100 кездейсоқ тордың бәрінде құны UCS-пен тең, ашылған түйіндері артық емес."""
    for seed in range(100):
        layout, pm = random_world(seed)
        alpha = [0.0, 0.3, 0.7, 1.0][seed % 4]
        cm = search.step_cost_map(pm, alpha)
        u = search.ucs(layout, cm, (0, 0), (7, 7))
        a = search.astar(layout, cm, (0, 0), (7, 7))
        assert u.found == a.found
        if u.found:
            assert a.cost == pytest.approx(u.cost)
            assert a.expanded <= u.expanded


def test_bfs_gives_fewest_steps():
    layout, pm = random_world(3)
    cm = search.step_cost_map(pm, 1.0)
    b = search.bfs(layout, cm, (0, 0), (7, 7))
    u = search.ucs(layout, cm, (0, 0), (7, 7))
    if b.found:
        assert b.steps <= u.steps


def test_alpha_zero_is_shortest_path():
    layout, pm = random_world(5)
    r = search.run("A*", layout, pm, 0.0, (0, 0), (7, 7))
    b = search.run("BFS", layout, pm, 0.0, (0, 0), (7, 7))
    if r.found:
        assert r.steps == b.steps


def test_clean_route_avoids_polluted_row():
    layout = np.zeros((8, 8), dtype=int)
    pm = np.full((8, 8), 20.0)
    pm[3, 1:7] = 150.0  # лас ойық: тіке жол соның ортасынан өтеді
    short = search.run("A*", layout, pm, 0.0, (0, 3), (7, 3))
    clean = search.run("A*", layout, pm, 1.0, (0, 3), (7, 3))
    dose = lambda res: sum(pm[r, c] for r, c in res.path[1:])
    assert clean.steps >= short.steps
    assert dose(clean) < dose(short)


def test_no_path_returns_empty_result():
    layout = np.zeros((8, 8), dtype=int)
    layout[:, 4] = C.BUILDING  # тор екіге бөлінген
    pm = np.full((8, 8), 30.0)
    for algo in ("BFS", "UCS", "A*"):
        res = search.run(algo, layout, pm, 0.5, (0, 0), (0, 7))
        assert not res.found and res.path == []


def test_blocked_cells_are_never_crossed():
    layout = np.zeros((8, 8), dtype=int)
    pm = np.full((8, 8), 30.0)
    blocked = {(3, 3), (3, 4), (4, 3)}
    res = search.run("A*", layout, pm, 0.5, (0, 0), (7, 7), blocked)
    assert res.found and not blocked & set(res.path)


@pytest.mark.parametrize("start,goal", [((0, 2), (7, 7)), ((0, 0), (1, 1)), ((9, 9), (0, 0)), ((0, 0), (-1, 3))])
def test_invalid_endpoints_raise(start, goal):
    layout = np.zeros((8, 8), dtype=int)
    layout[0, 2] = layout[1, 1] = C.BUILDING
    with pytest.raises(ValueError):
        search.run("A*", layout, np.full((8, 8), 30.0), 0.5, start, goal)


def test_start_equals_goal():
    layout = np.zeros((8, 8), dtype=int)
    res = search.run("A*", layout, np.full((8, 8), 30.0), 0.5, (2, 2), (2, 2))
    assert res.found and res.steps == 0 and res.cost == 0


def test_invalid_alpha_and_algorithm():
    layout = np.zeros((8, 8), dtype=int)
    pm = np.full((8, 8), 30.0)
    with pytest.raises(ValueError):
        search.run("A*", layout, pm, 1.5, (0, 0), (1, 1))
    with pytest.raises(ValueError):
        search.run("DFS", layout, pm, 0.5, (0, 0), (1, 1))
    with pytest.raises(ValueError):
        search.run("A*", layout, pm, 0.5, (0, 0), (1, 1), {(0, 0)})
