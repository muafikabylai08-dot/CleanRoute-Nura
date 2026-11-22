import numpy as np

import config as C
from src.hmm import HMM, classify, discretize
from src.pipeline import fit_hmm
from src.simulator import simulate


def small_hmm():
    h = HMM(n_states=2, n_obs=2)
    h.pi = np.array([0.6, 0.4])
    h.A = np.array([[0.7, 0.3], [0.4, 0.6]])
    h.B = np.array([[0.9, 0.1], [0.2, 0.8]])
    return h


def test_forward_rows_sum_to_one():
    h = small_hmm()
    alpha = h.forward([0, 1, 1, 0, 1])
    assert np.allclose(alpha.sum(axis=1), 1.0)


def test_viterbi_matches_brute_force():
    """Қысқа тізбекте Viterbi барлық 2^T жолдың ішіндегі ең ықтималын табады."""
    import itertools
    h = small_hmm()
    obs = [0, 1, 1, 0, 1, 1]

    def logp(path):
        p = np.log(h.pi[path[0]]) + np.log(h.B[path[0], obs[0]])
        for t in range(1, len(obs)):
            p += np.log(h.A[path[t - 1], path[t]]) + np.log(h.B[path[t], obs[t]])
        return p

    best = max(itertools.product([0, 1], repeat=len(obs)), key=logp)
    assert list(h.viterbi(obs)) == list(best)


def test_viterbi_beats_plain_threshold_on_noisy_sensors():
    """Симулятордан шын күйлер белгілі: HMM шулы оқылымды тікелей шекпен бөлуден дәлірек."""
    d = simulate()
    n_train = int(len(d.hours) * 0.8)
    h = fit_hmm(d, n_train)
    acc_thr, acc_vit = [], []
    for i, (r, c) in enumerate(d.sensor_cells):
        true = classify(d.pm_true[n_train:, r, c])
        raw = d.readings[n_train:, i]
        acc_thr.append(np.mean(classify(raw) == true))
        acc_vit.append(np.mean(h.viterbi(discretize(raw)) == true))
    assert np.mean(acc_vit) > np.mean(acc_thr) + 0.05


def test_classify_thresholds():
    assert list(classify([10, C.PM_LOW, C.PM_LOW + 0.1, C.PM_HIGH, C.PM_HIGH + 0.1])) == [0, 0, 1, 1, 2]


def test_discretize_range():
    obs = discretize([0, 14.9, 15, 129, 130, 999])
    assert obs.min() >= 0 and obs.max() <= len(C.OBS_BIN_EDGES) - 1
