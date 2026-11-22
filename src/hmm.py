"""7-апта: Hidden Markov Model.

Жасырын күй - ұяшықтағы ауаның нақты ластану класы (Таза / Орташа / Қауіпті).
Бақылау - датчиктің шулы оқылымы (дискреттелген қорапшалар).

Forward - әр сәттегі күй сенімі (filtering), Viterbi - ең ықтимал күйлер тізбегі.
Параметрлер (бастапқы, ауысу, эмиссия) белгіленген деректерден санау арқылы бағаланады
(супервизиялы, Лаплас тегістеуімен).
"""
from __future__ import annotations

import numpy as np

import config as C


def classify(pm) -> np.ndarray:
    """PM2.5 -> күй индексі (0 Таза, 1 Орташа, 2 Қауіпті)."""
    pm = np.asarray(pm, dtype=float)
    return np.where(pm <= C.PM_LOW, 0, np.where(pm <= C.PM_HIGH, 1, 2))


def discretize(readings) -> np.ndarray:
    """Оқылымды қорапша индексіне айналдырады."""
    return np.digitize(np.asarray(readings, dtype=float), C.OBS_BIN_EDGES[1:])


class HMM:
    def __init__(self, n_states: int = 3, n_obs: int = len(C.OBS_BIN_EDGES)):
        self.n_states, self.n_obs = n_states, n_obs
        self.pi = np.full(n_states, 1 / n_states)
        self.A = np.full((n_states, n_states), 1 / n_states)
        self.B = np.full((n_states, n_obs), 1 / n_obs)
        self.state_pm_mean = np.array([25.0, 50.0, 90.0])

    def fit_supervised(self, state_seqs, obs_seqs, pm_seqs=None, smoothing: float = 1.0) -> "HMM":
        pi = np.full(self.n_states, smoothing)
        A = np.full((self.n_states, self.n_states), smoothing)
        B = np.full((self.n_states, self.n_obs), smoothing)
        for s, o in zip(state_seqs, obs_seqs):
            pi[s[0]] += 1
            for a, b in zip(s[:-1], s[1:]):
                A[a, b] += 1
            for st, ob in zip(s, o):
                B[st, ob] += 1
        self.pi = pi / pi.sum()
        self.A = A / A.sum(axis=1, keepdims=True)
        self.B = B / B.sum(axis=1, keepdims=True)
        if pm_seqs is not None:
            allst = np.concatenate(state_seqs)
            allpm = np.concatenate(pm_seqs)
            for k in range(self.n_states):
                if (allst == k).any():
                    self.state_pm_mean[k] = allpm[allst == k].mean()
        return self

    def forward(self, obs) -> np.ndarray:
        """Нормаланған filtering сенімдері, пішіні (T, n_states). Әр қатардың қосындысы = 1."""
        obs = np.asarray(obs, dtype=int)
        alpha = np.zeros((len(obs), self.n_states))
        a = self.pi * self.B[:, obs[0]]
        alpha[0] = a / a.sum()
        for t in range(1, len(obs)):
            a = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            alpha[t] = a / a.sum()
        return alpha

    def viterbi(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        logA, logB = np.log(self.A), np.log(self.B)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = np.log(self.pi) + logB[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[:, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def expected_pm(self, beliefs: np.ndarray) -> np.ndarray:
        """Күй сенімдерінен күтілетін PM2.5."""
        return beliefs @ self.state_pm_mean
