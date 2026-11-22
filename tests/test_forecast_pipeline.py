import numpy as np
import pytest

import config as C
from src.agent import RouteAgent, apply_overrides, performance
from src.pipeline import idw_grid
from src.simulator import load_readings, simulate


@pytest.fixture(scope="module")
def agent():
    return RouteAgent(simulate(hours=24 * 14))


def test_idw_reproduces_sensor_values():
    cells = [(0, 0), (7, 7)]
    g = idw_grid(cells, [10.0, 90.0])
    assert g[0, 0] == pytest.approx(10.0, abs=1e-3) and g[7, 7] == pytest.approx(90.0, abs=1e-3)
    assert 10 <= g.min() and g.max() <= 90


def test_forecast_shape_and_nonnegative(agent):
    for h in (1, 2, 3):
        m = agent.pm_map(200, h)
        assert m.shape == (8, 8) and np.all(m >= 0) and np.isfinite(m).all()


def test_forecast_rmse_beats_persistence(agent):
    ev = agent.forecaster.evaluate(2)
    assert ev["Random Forest"]["RMSE"] < ev["Базалық (өзгермейді)"]["RMSE"]


def test_forecast_uses_only_past_data(agent):
    """Болжам t уақытынан кейінгі деректерге тәуелді емес: болашақты бұзсақ, нәтиже өзгермейді."""
    t = 150
    before = agent.pm_map(t, 1).copy()
    saved = agent.grid[t + 1:].copy()
    agent.grid[t + 1:] = 999.0
    try:
        after = agent.pm_map(t, 1)
    finally:
        agent.grid[t + 1:] = saved
    assert np.allclose(before, after)


def test_invalid_forecast_requests(agent):
    with pytest.raises(ValueError):
        agent.pm_map(0, 1)       # тарих жеткіліксіз
    with pytest.raises(ValueError):
        agent.pm_map(10_000, 0)  # уақыт диапазоннан тыс


@pytest.mark.parametrize("bad", [-5, 5000, float("nan"), "abc", None])
def test_invalid_override_values(bad):
    with pytest.raises(ValueError):
        apply_overrides(np.full((8, 8), 30.0), {(1, 1): bad})


def test_override_applies_and_does_not_mutate():
    pm = np.full((8, 8), 30.0)
    out = apply_overrides(pm, {(1, 1): 120})
    assert out[1, 1] == 120 and pm[1, 1] == 30.0


def test_plan_and_explain(agent):
    plan = agent.plan(200, 1, (0, 0), (7, 7), 1.0, "A*")
    assert plan["result"].found
    # alpha = 1: құн = доза / pm_ref, сондықтан оңтайлы жолдың дозасы қысқа жолдан көп емес
    assert plan["perf"]["dose"] <= plan["perf_shortest"]["dose"] + 1e-9
    assert "қадам" in agent.explain(plan, 1.0)


def test_performance_empty_path():
    assert performance([], np.full((8, 8), 30.0))["steps"] == 0


def test_load_readings_validation(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("hour,wind_dr\n1,2\n")
    with pytest.raises(ValueError):
        load_readings(bad)
    cols = "hour,wind_dr,wind_dc,wind_speed," + ",".join(f"s{i}" for i in range(len(C.SENSOR_CELLS)))
    neg = tmp_path / "neg.csv"
    neg.write_text(cols + "\n0,0,1,0.2," + ",".join(["-5"] * len(C.SENSOR_CELLS)) + "\n")
    with pytest.raises(ValueError):
        load_readings(neg)
    txt = tmp_path / "txt.csv"
    txt.write_text(cols + "\n0,0,1,0.2," + ",".join(["abc"] * len(C.SENSOR_CELLS)) + "\n")
    with pytest.raises(ValueError):
        load_readings(txt)


def test_csv_mode_end_to_end(tmp_path):
    """readings.csv режимі (шын өріссіз): сақтау -> оқу -> агент -> маршрут."""
    from src.simulator import save_data
    sim = simulate(hours=24 * 14)
    rp, gp = tmp_path / "readings.csv", tmp_path / "grid.json"
    save_data(sim, rp, gp)
    loaded = load_readings(rp)
    assert loaded.pm_true is None and loaded.readings.shape == sim.readings.shape
    agent = RouteAgent(loaded)
    plan = agent.plan(150, 2, (0, 0), (7, 7), 0.8, "A*")
    assert plan["result"].found
