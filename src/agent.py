"""1-апта: интеллектуалды агент.

PEAS:
  Performance - маршруттағы ластану дозасы (аз - жақсы) және жол ұзындығы;
  Environment - Нұра ықшамауданының 8x8 торы, ластану өрісі уақыт бойынша өзгереді;
  Actuators   - маршрут (ұяшықтар тізбегі) және түсіндірме;
  Sensors     - ауа датчиктері (шулы), жел және сағат.

Агент "мақсатқа негізделген + пайдалылыққа негізделген": қабылдайды (HMM + карта), болжайды (ML)
және жоспарлайды (A*).
"""
from __future__ import annotations

import math

import numpy as np

import config as C
from src import search
from src.forecast import Forecaster
from src.hmm import classify
from src.pipeline import estimate_grid, fit_hmm
from src.simulator import SimData


def apply_overrides(pm: np.ndarray, overrides: dict | None) -> np.ndarray:
    """Қолмен енгізілген PM2.5 мәндерін тексеріп, картаға қолданады."""
    out = pm.copy()
    for (r, c), v in (overrides or {}).items():
        if not (0 <= r < C.GRID_SIZE and 0 <= c < C.GRID_SIZE):
            raise ValueError(f"Ұяшық тордан тыс: {(r, c)}")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError("PM2.5 мәні сан болуы керек")
        if not math.isfinite(v) or v < 0 or v > C.MAX_VALID_READING:
            raise ValueError(f"PM2.5 мәні 0..{C.MAX_VALID_READING:g} аралығында болуы керек")
        out[r, c] = v
    return out


def performance(path, pm: np.ndarray) -> dict:
    """Маршрутты бағалау: қадам, ластану дозасы (бастау ұяшығынан басқа), қауіпті ұяшықтар."""
    cells = path[1:]
    vals = np.array([pm[r, c] for r, c in cells]) if cells else np.array([])
    return {
        "steps": max(len(path) - 1, 0),
        "dose": float(vals.sum()),
        "mean_pm": float(vals.mean()) if len(vals) else 0.0,
        "danger_cells": int((classify(vals) == 2).sum()) if len(vals) else 0,
    }


class RouteAgent:
    def __init__(self, data: SimData, train_frac: float = 0.8):
        self.data = data
        self.layout = data.layout
        n_train = int(len(data.hours) * train_frac)
        self.hmm = fit_hmm(data, n_train)
        self.beliefs, self.sensor_est, self.grid = estimate_grid(data, self.hmm)
        self.forecaster = Forecaster()
        self.forecaster.set_context(self.grid, data.hours, data.wind, self.layout,
                                    truth=data.pm_true, train_frac=train_frac)

    # қабылдау және болжау
    def pm_map(self, t: int, horizon: int = 0, overrides: dict | None = None) -> np.ndarray:
        T = len(self.data.hours)
        if not 0 <= t < T:
            raise ValueError("Уақыт индексі диапазоннан тыс")
        pm = self.forecaster.predict(t, horizon)
        return apply_overrides(pm, overrides)

    # жоспарлау
    def plan(self, t, horizon, start, goal, alpha, algorithm, blocked=None, overrides=None) -> dict:
        pm = self.pm_map(t, horizon, overrides)
        result = search.run(algorithm, self.layout, pm, alpha, start, goal, blocked)
        shortest = search.run("A*", self.layout, pm, 0.0, start, goal, blocked)
        return {
            "pm": pm,
            "result": result,
            "shortest": shortest,
            "perf": performance(result.path, pm) if result.found else None,
            "perf_shortest": performance(shortest.path, pm) if shortest.found else None,
        }

    def compare(self, pm, alpha, start, goal, blocked=None) -> list:
        return [search.run(a, self.layout, pm, alpha, start, goal, blocked) for a in search.ALGORITHMS]

    @staticmethod
    def explain(plan: dict, alpha: float) -> str:
        res, sh = plan["result"], plan["shortest"]
        if not res.found:
            return "Жол табылмады: бастау мен аяқтау нүктесі тосқауылдармен бөлінген."
        p, ps = plan["perf"], plan["perf_shortest"]
        txt = (f"Таңдалған маршрут: {p['steps']} қадам, ластану дозасы {p['dose']:.0f} "
               f"(орташа PM2.5 {p['mean_pm']:.0f}), қауіпті ұяшық саны {p['danger_cells']}.")
        if alpha == 0:
            return txt + " alpha = 0, сондықтан бұл ең қысқа маршрут."
        d_steps = p["steps"] - ps["steps"]
        if ps["dose"] > 0:
            d_dose = (p["dose"] - ps["dose"]) / ps["dose"] * 100
        else:
            d_dose = 0.0
        if d_steps == 0 and abs(d_dose) < 0.5:
            return txt + " Ең қысқа маршрутпен сәйкес келеді: айналып өтетін таза жол жоқ немесе ол тым ұзақ."
        return txt + (f" Ең қысқа маршрутпен салыстырғанда: {d_steps:+d} қадам, ластану дозасы {d_dose:+.0f}%, "
                      f"қауіпті ұяшықтар {p['danger_cells']} (қысқа жолда {ps['danger_cells']}).")
