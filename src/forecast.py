"""8-апта: Classical ML. Келесі сағаттардағы PM2.5 өрісін болжау.

Әр (уақыт, ұяшық) жұбы - бір үлгі. Белгілер: ұяшықтың өткен мәндері, төрт көршінің мәні,
жел, тәулік сағаты, ұяшық түрі, орны. Мақсат - h сағаттан кейінгі PM2.5.
Модельдер: Ridge (сызықтық) және Random Forest. Базалық модель: "өзгермейді" (persistence).
Бөлу хронологиялық: алғашқы 80% оқыту, соңғы 20% тест (болашақты көрмейді).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from src.hmm import classify

N_LAGS = 2  # t-1, t-2


def build_features(grid: np.ndarray, hours, wind, layout, ts) -> np.ndarray:
    """grid: (T,8,8) бағаланған карта. ts: уақыт индекстері (n,), ts >= N_LAGS. -> (n*64, F)."""
    ts = np.asarray(ts, dtype=int)
    n = len(ts)
    cur = grid[ts]
    lag1, lag2 = grid[ts - 1], grid[ts - 2]
    p = np.pad(cur, ((0, 0), (1, 1), (1, 1)), mode="edge")
    north, south = p[:, :-2, 1:-1], p[:, 2:, 1:-1]
    west, east = p[:, 1:-1, :-2], p[:, 1:-1, 2:]
    h = np.asarray(hours)[ts]
    hs = np.broadcast_to(np.sin(2 * np.pi * h / 24)[:, None, None], cur.shape)
    hc = np.broadcast_to(np.cos(2 * np.pi * h / 24)[:, None, None], cur.shape)
    w = np.asarray(wind)[ts]
    wr, wc, ws = (np.broadcast_to(w[:, i][:, None, None], cur.shape) for i in range(3))
    types = np.broadcast_to(layout[None], cur.shape)
    onehot = [(types == k).astype(float) for k in (C.ROAD, C.BUILDING, C.PARK, C.MAIN_ROAD)]
    rr, cc = np.mgrid[0:C.GRID_SIZE, 0:C.GRID_SIZE]
    rowf = np.broadcast_to(rr[None], cur.shape)
    colf = np.broadcast_to(cc[None], cur.shape)
    feats = [cur, lag1, lag2, north, south, west, east, hs, hc, wr, wc, ws, *onehot, rowf, colf]
    return np.stack([f.reshape(n * C.GRID_SIZE ** 2) for f in feats], axis=1)


def _metrics(y_true_pm, y_pred_pm) -> dict:
    yt, yp = classify(y_true_pm), classify(y_pred_pm)
    return {
        "MAE": float(mean_absolute_error(y_true_pm, y_pred_pm)),
        "RMSE": float(np.sqrt(np.mean((y_true_pm - y_pred_pm) ** 2))),
        "Accuracy": float(np.mean(yt == yp)),
        "macro-F1": float(f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)),
    }


class Forecaster:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.models: dict[tuple[int, str], object] = {}
        self.ctx = None

    def _make(self, kind: str):
        if kind == "rf":
            return RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=5,
                                         random_state=self.seed, n_jobs=-1)
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    def set_context(self, grid, hours, wind, layout, truth=None, train_frac: float = 0.8):
        self.ctx = dict(grid=grid, hours=hours, wind=wind, layout=layout, truth=truth, train_frac=train_frac)
        self.models.clear()

    def _split(self, horizon: int):
        T = self.ctx["grid"].shape[0]
        ts = np.arange(N_LAGS, T - horizon)
        cut = int(len(ts) * self.ctx["train_frac"])
        return ts[:cut], ts[cut:]

    def _xy(self, ts, horizon: int):
        c = self.ctx
        X = build_features(c["grid"], c["hours"], c["wind"], c["layout"], ts)
        y = c["grid"][ts + horizon].reshape(-1)
        return X, y

    def ensure(self, horizon: int):
        """h сағат үшін екі модельді үйретеді (ұсақ: бір рет)."""
        if horizon < 1:
            raise ValueError("Болжам сағаты >= 1 болуы керек")
        if (horizon, "rf") in self.models:
            return
        tr, _ = self._split(horizon)
        X, y = self._xy(tr, horizon)
        for kind in ("rf", "ridge"):
            self.models[(horizon, kind)] = self._make(kind).fit(X, y)

    def evaluate(self, horizon: int) -> dict:
        """Тест бөлігіндегі метрикалар: Ridge, RF және базалық модель (+ шын өріспен салыстыру)."""
        self.ensure(horizon)
        _, te = self._split(horizon)
        X, y = self._xy(te, horizon)
        base = self.ctx["grid"][te].reshape(-1)
        out = {"Базалық (өзгермейді)": _metrics(y, base)}
        preds = {}
        for kind, name in (("ridge", "Ridge"), ("rf", "Random Forest")):
            preds[kind] = self.models[(horizon, kind)].predict(X)
            out[name] = _metrics(y, preds[kind])
        out["_confusion_rf"] = confusion_matrix(classify(y), classify(preds["rf"]), labels=[0, 1, 2]).tolist()
        truth = self.ctx["truth"]
        if truth is not None:
            yt = classify(truth[te + horizon].reshape(-1))
            out["_true_acc"] = {
                "Базалық (өзгермейді)": float(np.mean(classify(base) == yt)),
                "Random Forest": float(np.mean(classify(preds["rf"]) == yt)),
            }
        out["_test_size"] = int(len(te))
        return out

    def predict(self, t: int, horizon: int, kind: str = "rf") -> np.ndarray:
        """t уақытынан h сағат кейінгі PM2.5 өрісі (8x8)."""
        c = self.ctx
        if horizon == 0:
            return c["grid"][t].copy()
        if t < N_LAGS:
            raise ValueError(f"Болжам үшін кемінде {N_LAGS} сағат тарих керек")
        self.ensure(horizon)
        X = build_features(c["grid"], c["hours"], c["wind"], c["layout"], [t])
        return np.maximum(self.models[(horizon, kind)].predict(X), 0).reshape(C.GRID_SIZE, C.GRID_SIZE)
