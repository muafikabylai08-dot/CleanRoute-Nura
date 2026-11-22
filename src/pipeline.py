"""Деректерді өңдеу тізбегі: датчик -> HMM -> күтілетін PM2.5 -> 8x8 карта (интерполяция)."""
from __future__ import annotations

import numpy as np

import config as C
from src.hmm import HMM, classify, discretize
from src.simulator import SimData


def idw_grid(sensor_cells, values, power: float = 2.0) -> np.ndarray:
    """Қашықтыққа кері пропорционал салмақтау. values: (S,) немесе (T, S) -> (8,8) немесе (T,8,8)."""
    n = C.GRID_SIZE
    rr, cc = np.mgrid[0:n, 0:n]
    w = np.zeros((len(sensor_cells), n, n))
    for i, (r, c) in enumerate(sensor_cells):
        d2 = (rr - r) ** 2 + (cc - c) ** 2
        w[i] = 1.0 / np.maximum(d2, 1e-6) ** (power / 2)
    w /= w.sum(axis=0, keepdims=True)
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        return np.tensordot(vals, w, axes=(0, 0))
    return np.tensordot(vals, w, axes=(1, 0))


def fit_hmm(data: SimData, n_train: int) -> HMM:
    """HMM-ді алғашқы n_train сағатта үйретеді.
    Шын өріс белгілі болса (симулятор), шын кластарды қолданады; әйтпесе оқылымдарды
    жылжымалы орташамен тегістеп, жалған белгілер жасайды."""
    state_seqs, obs_seqs, pm_seqs = [], [], []
    for i, (r, c) in enumerate(data.sensor_cells):
        raw = data.readings[:n_train, i]
        if data.pm_true is not None:
            clean = data.pm_true[:n_train, r, c]
        else:
            k = 5
            clean = np.convolve(np.pad(raw, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        state_seqs.append(classify(clean))
        obs_seqs.append(discretize(raw))
        pm_seqs.append(clean)
    return HMM().fit_supervised(state_seqs, obs_seqs, pm_seqs)


def estimate_grid(data: SimData, hmm: HMM):
    """Әр сағат үшін: датчик күй сенімдері, күтілетін PM2.5 және толық 8x8 карта."""
    T, S = data.readings.shape
    beliefs = np.zeros((T, S, hmm.n_states))
    est = np.zeros((T, S))
    for i in range(S):
        b = hmm.forward(discretize(data.readings[:, i]))
        beliefs[:, i, :] = b
        est[:, i] = hmm.expected_pm(b)
    grid = idw_grid(data.sensor_cells, est)
    return beliefs, est, grid
