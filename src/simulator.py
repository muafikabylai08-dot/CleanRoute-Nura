"""Синтетикалық деректер генераторы.

Нақты датчиктер аз болса, жүйені тексеру үшін қолданылады. Артықшылығы: ластанудың
"шын" өрісі белгілі, сондықтан HMM мен ML дәлдігін объективті өлшеуге болады.

Модель: әр сағатта PM2.5 өрісі (8x8) желмен жылжиды (upwind), көршілерге таралады,
ішінара сөнеді және көздерден (магистраль, ғимараттар) толықтырылады.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C

# жел бағыты: (dr, dc) - ауа қай жаққа қозғалады: 0 - солтүстік, 1 - шығыс, 2 - оңтүстік, 3 - батыс
WIND_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@dataclass
class SimData:
    layout: np.ndarray        # (8, 8) ұяшық түрлері
    hours: np.ndarray         # (T,) тәулік сағаты 0..23
    wind: np.ndarray          # (T, 3): dr, dc, жылдамдық
    readings: np.ndarray      # (T, S) шулы датчик оқылымдары
    sensor_cells: list
    pm_true: np.ndarray | None = None  # (T, 8, 8) шын өріс (тек симулятор)


def load_layout(path=C.GRID_PATH) -> np.ndarray:
    """grid.json файлын оқиды; жоқ болса, әдепкі тор қайтарады."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        layout = np.array(data["layout"], dtype=int)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        layout = np.array(C.DEFAULT_LAYOUT, dtype=int)
    if layout.shape != (C.GRID_SIZE, C.GRID_SIZE):
        raise ValueError(f"Тор өлшемі {C.GRID_SIZE}x{C.GRID_SIZE} болуы керек, алынды {layout.shape}")
    return layout


def _upwind(pm: np.ndarray, d: tuple[int, int]) -> np.ndarray:
    """Ауа келетін көршінің мәні (шетте - ұяшықтың өз мәні)."""
    dr, dc = d
    up = np.pad(pm, 1, mode="edge")
    n = pm.shape[0]
    return up[1 - dr:1 - dr + n, 1 - dc:1 - dc + n]


def _laplacian(pm: np.ndarray) -> np.ndarray:
    p = np.pad(pm, 1, mode="edge")
    avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
    return avg - pm


def _emission(layout: np.ndarray, hour: int) -> np.ndarray:
    traffic = np.exp(-((hour - 8) ** 2) / 8) + np.exp(-((hour - 18) ** 2) / 8)
    night = 1.0 if (hour >= 18 or hour <= 6) else 0.3
    em = np.full(layout.shape, 0.5)
    em[layout == C.MAIN_ROAD] = 3.0 + 7.0 * traffic
    em[layout == C.ROAD] = 0.8 + 1.5 * traffic
    em[layout == C.BUILDING] = 1.0 + 3.0 * night   # жылыту көздері
    em[layout == C.PARK] = 0.1
    return em


def simulate(hours: int = C.SIM_HOURS, seed: int = C.SEED, layout: np.ndarray | None = None) -> SimData:
    rng = np.random.default_rng(seed)
    layout = load_layout() if layout is None else layout
    pm = np.full(layout.shape, 25.0)
    background = 5.0
    wdir = int(rng.integers(0, 4))
    wspeed = 0.2

    pm_hist = np.zeros((hours, *layout.shape))
    wind = np.zeros((hours, 3))
    hr = np.arange(hours) % 24
    for t in range(hours):
        if rng.random() < 0.04:
            wdir = int(rng.integers(0, 4))
        wspeed = float(np.clip(0.9 * wspeed + 0.1 * 0.2 + rng.normal(0, 0.05), 0.03, 0.45))
        background = float(np.clip(0.96 * background + rng.normal(0, 1.8), 0.0, 40.0))
        d = WIND_DIRS[wdir]
        pm = (1 - wspeed) * pm + wspeed * _upwind(pm, d)
        pm = pm + 0.12 * _laplacian(pm)
        pm = 0.90 * pm + _emission(layout, int(hr[t])) * 1.9 + background * 0.15
        pm = np.maximum(pm, 1.0)
        pm_hist[t] = pm
        wind[t] = (d[0], d[1], wspeed)

    sensors = list(C.SENSOR_CELLS)
    clean = np.stack([pm_hist[:, r, c] for r, c in sensors], axis=1)
    readings = np.maximum(clean + rng.normal(0, C.SENSOR_NOISE, clean.shape), 0.0)
    return SimData(layout, hr, wind, readings, sensors, pm_hist)


def save_data(sim: SimData, readings_path=C.READINGS_PATH, grid_path=C.GRID_PATH) -> None:
    readings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump({"size": C.GRID_SIZE, "layout": sim.layout.tolist(),
                   "sensors": [list(s) for s in sim.sensor_cells]}, f, ensure_ascii=False, indent=1)
    df = pd.DataFrame({"hour": sim.hours, "wind_dr": sim.wind[:, 0],
                       "wind_dc": sim.wind[:, 1], "wind_speed": sim.wind[:, 2]})
    for i in range(sim.readings.shape[1]):
        df[f"s{i}"] = sim.readings[:, i].round(1)
    df.to_csv(readings_path, index=False)


def load_readings(path=C.READINGS_PATH) -> SimData:
    """readings.csv оқиды және тексереді. Файл жоқ/бұзылған болса, қате көтереді."""
    df = pd.read_csv(path)
    n_s = len(C.SENSOR_CELLS)
    needed = ["hour", "wind_dr", "wind_dc", "wind_speed"] + [f"s{i}" for i in range(n_s)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"readings.csv ішінде бағандар жоқ: {missing}")
    vals = df[needed].apply(pd.to_numeric, errors="coerce")
    if vals.isna().any().any():
        raise ValueError("readings.csv ішінде сан емес немесе бос мәндер бар")
    sens = vals[[f"s{i}" for i in range(n_s)]].to_numpy()
    if (sens < 0).any() or (sens > C.MAX_VALID_READING).any():
        raise ValueError(f"Датчик оқылымы 0..{C.MAX_VALID_READING:g} аралығында болуы керек")
    return SimData(load_layout(), vals["hour"].to_numpy(int) % 24,
                   vals[["wind_dr", "wind_dc", "wind_speed"]].to_numpy(), sens,
                   list(C.SENSOR_CELLS), None)


def get_data(seed: int = C.SEED) -> tuple[SimData, str]:
    """Файл болса - файлдан, болмаса/бұзылса - симулятордан. Дереккөз атын да қайтарады."""
    if C.READINGS_PATH.exists():
        try:
            return load_readings(), "readings.csv"
        except (ValueError, OSError, pd.errors.ParserError):
            return simulate(seed=seed), "симулятор (readings.csv жарамсыз)"
    return simulate(seed=seed), "симулятор"


if __name__ == "__main__":
    s = simulate()
    save_data(s)
    print(f"Жазылды: {C.GRID_PATH}, {C.READINGS_PATH}")
