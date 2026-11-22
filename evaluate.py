"""Барлық метрикаларды қайта есептейді (Technical Report мен README үшін).
Іске қосу: python evaluate.py
"""
import numpy as np

import config as C
from src import search
from src.agent import RouteAgent
from src.hmm import classify, discretize
from src.simulator import simulate


def main(seed: int = C.SEED):
    d = simulate(seed=seed)
    agent = RouteAgent(d)
    T = len(d.hours)
    n_train = int(T * 0.8)
    print(f"Симулятор: seed={seed}, {T} сағат, оқыту {n_train}, тест {T - n_train}\n")

    print("== HMM (тест бөлігі, 6 датчиктің орташасы, шын күйге қарағанда дәлдік) ==")
    acc = {"Шек (шулы оқылым)": [], "Forward": [], "Viterbi": []}
    for i, (r, c) in enumerate(d.sensor_cells):
        true = classify(d.pm_true[n_train:, r, c])
        raw = d.readings[n_train:, i]
        obs = discretize(raw)
        acc["Шек (шулы оқылым)"].append(np.mean(classify(raw) == true))
        acc["Forward"].append(np.mean(agent.hmm.forward(obs).argmax(1) == true))
        acc["Viterbi"].append(np.mean(agent.hmm.viterbi(obs) == true))
    for k, v in acc.items():
        print(f"  {k:20s} {np.mean(v):.3f}")

    print("\n== ML болжам (тест, соңғы 20%) ==")
    for h in (1, 2, 3):
        ev = agent.forecaster.evaluate(h)
        print(f"  h={h} сағат, тест қадамдары: {ev['_test_size']}")
        for name in ("Базалық (өзгермейді)", "Ridge", "Random Forest"):
            m = ev[name]
            print(f"    {name:22s} MAE {m['MAE']:.2f}  RMSE {m['RMSE']:.2f}  Acc {m['Accuracy']:.3f}  macro-F1 {m['macro-F1']:.3f}")
        ta = ev["_true_acc"]
        print(f"    шын өріспен класс дәлдігі: базалық {ta['Базалық (өзгермейді)']:.3f}, RF {ta['Random Forest']:.3f}")

    print("\n== Маршрут: таза (alpha=1) және ең қысқа (alpha=0) A* салыстыруы ==")
    rng = np.random.default_rng(1)
    free = [(r, c) for r in range(8) for c in range(8) if d.layout[r, c] != C.BUILDING]
    red, extra, differ = [], [], []
    for _ in range(400):
        s, g = [free[i] for i in rng.choice(len(free), 2, replace=False)]
        if abs(s[0] - g[0]) + abs(s[1] - g[1]) < 6:
            continue
        t = int(rng.integers(n_train + 2, T - 4))
        pm = agent.pm_map(t, 1)
        sh = search.run("A*", d.layout, pm, 0.0, s, g)
        cl = search.run("A*", d.layout, pm, 1.0, s, g)
        ds = sum(pm[r, c] for r, c in sh.path[1:])
        dc = sum(pm[r, c] for r, c in cl.path[1:])
        red.append((dc - ds) / ds * 100)
        extra.append(cl.steps - sh.steps)
        differ.append(cl.path != sh.path)
    red = np.array(red)
    print(f"  сынақ саны: {len(red)} (кездейсоқ нүктелер, Манхэттен қашықтығы >= 6, 1 сағаттық болжам)")
    print(f"  доза өзгерісі: орташа {red.mean():.1f}%, медиана {np.median(red):.1f}%, ең жақсы {red.min():.1f}%")
    print(f"  қосымша қадам: орташа {np.mean(extra):.2f}, ең көбі {max(extra)}")
    print(f"  таза маршрут ең қысқадан өзгеше: {np.mean(differ) * 100:.0f}% жағдайда")

    print("\n== Алгоритмдер (A* vs UCS, 100 кездейсоқ ластану картасы, alpha=1) ==")
    ratios = []
    for k in range(100):
        pm = np.random.default_rng(k).uniform(10, 120, (8, 8))
        cm = search.step_cost_map(pm, 1.0)
        u = search.ucs(d.layout, cm, (0, 0), (7, 7))
        a = search.astar(d.layout, cm, (0, 0), (7, 7))
        ratios.append(a.expanded / u.expanded)
    print(f"  A* ашқан түйіндер / UCS ашқан түйіндер: орташа {np.mean(ratios):.2f}")


if __name__ == "__main__":
    main()
