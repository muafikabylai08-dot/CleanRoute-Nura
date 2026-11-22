"""CleanRoute Nura: Streamlit интерфейсі. Іске қосу: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from src import search
from src.agent import RouteAgent
from src.hmm import discretize
from src.simulator import get_data

st.set_page_config(page_title="CleanRoute Nura", layout="wide")
N = C.GRID_SIZE


@st.cache_resource(show_spinner="Деректер мен модельдер дайындалуда...")
def load_agent():
    data, source = get_data()
    return RouteAgent(data), source


@st.cache_data(show_spinner="ML бағалануда...")
def cached_eval(_agent: RouteAgent, horizon: int, key: str):
    return _agent.forecaster.evaluate(horizon)


def heat_figure(layout, pm, path=None, shortest=None, start=None, goal=None, blocked=(), explored=None, current=None):
    z = pm.astype(float)
    text = np.round(z).astype(int).astype(str)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=z, text=text, texttemplate="%{text}", zmin=10, zmax=130,
                             colorscale=[[0, "#2e9e5b"], [0.25, "#9bd16b"], [0.5, "#f2d04a"],
                                         [0.75, "#ef8a3a"], [1, "#c8372d"]],
                             colorbar=dict(title="PM2.5"), hovertemplate="қатар %{y}, баған %{x}<br>PM2.5 %{z:.0f}<extra></extra>"))
    wall = np.where(layout == C.BUILDING, 1.0, np.nan)
    for r, c in blocked:
        wall[r, c] = 1.0
    fig.add_trace(go.Heatmap(z=wall, colorscale=[[0, "#555"], [1, "#555"]], showscale=False,
                             hovertemplate="тосқауыл<extra></extra>"))
    for r, c in C.SENSOR_CELLS:
        fig.add_trace(go.Scatter(x=[c], y=[r], mode="markers", marker=dict(symbol="diamond", size=9, color="white",
                                 line=dict(color="black", width=1)), showlegend=False, hovertemplate="датчик<extra></extra>"))
    if explored:
        xs, ys = [c for r, c in explored], [r for r, c in explored]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[str(i + 1) for i in range(len(xs))],
                                 textfont=dict(size=9, color="black"),
                                 marker=dict(symbol="square-open", size=30, color="#1f4fd8", line=dict(width=2)),
                                 name="ашылған", hoverinfo="skip"))
    if shortest and (not path or shortest != path):
        fig.add_trace(go.Scatter(x=[c for r, c in shortest], y=[r for r, c in shortest], mode="lines",
                                 line=dict(color="black", width=3, dash="dash"), name="ең қысқа"))
    if path:
        fig.add_trace(go.Scatter(x=[c for r, c in path], y=[r for r, c in path], mode="lines+markers",
                                 line=dict(color="#1f4fd8", width=5), marker=dict(size=7), name="таңдалған маршрут"))
    if current:
        fig.add_trace(go.Scatter(x=[current[1]], y=[current[0]], mode="markers",
                                 marker=dict(symbol="star", size=20, color="#1f4fd8"), name="қазіргі түйін"))
    for pt, label, color in ((start, "Бастау", "#0b6b2f"), (goal, "Аяқтау", "#7a1fa2")):
        if pt:
            fig.add_trace(go.Scatter(x=[pt[1]], y=[pt[0]], mode="markers+text", text=[label], textposition="top center",
                                     marker=dict(size=16, color=color, symbol="circle", line=dict(color="white", width=2)),
                                     showlegend=False))
    fig.update_yaxes(autorange="reversed", tickmode="linear", dtick=1, title="қатар", scaleanchor="x")
    fig.update_xaxes(tickmode="linear", dtick=1, title="баған", side="top", constrain="domain")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.05))
    return fig


agent, source = load_agent()
T = len(agent.data.hours)

st.title("CleanRoute Nura")
st.caption("Нұра ықшамауданының 8×8 торында ауа ластануын ескеретін маршрут жоспарлағыш. "
           "Оқу демонстрациясы, медициналық кеңес емес.")

# ---------------- бүйір панелі ----------------
sb = st.sidebar
sb.header("Параметрлер")
sb.caption(f"Дерек көзі: {source}")
t = sb.slider("Уақыт (сағат индексі)", 2, T - 4, min(642, T - 4))
sb.caption(f"Тәулік сағаты: {int(agent.data.hours[t]):02d}:00")
horizon = sb.selectbox("Болжам", [0, 1, 2, 3], index=1, format_func=lambda h: "Қазіргі карта" if h == 0 else f"{h} сағаттан кейін")
algorithm = sb.selectbox("Алгоритм", list(search.ALGORITHMS), index=2)
alpha = sb.slider("Таза жол салмағы (alpha)", 0.0, 1.0, 1.0, 0.05,
                  help="0 - ең қысқа жол, 1 - ластануды барынша азайту")
cells = [(r, c) for r in range(N) for c in range(N)]
fmt = lambda x: f"({x[0]}, {x[1]})"
start = sb.selectbox("Бастау ұяшығы", cells, index=cells.index((0, 3)), format_func=fmt)
goal = sb.selectbox("Аяқтау ұяшығы", cells, index=cells.index((7, 4)), format_func=fmt)
extra_blocked = sb.multiselect("Қосымша тосқауыл ұяшықтар", cells, format_func=fmt)
blocked = set(extra_blocked)

overrides = {}
with sb.expander("Датчик оқылымын қолмен енгізу"):
    use_ov = st.checkbox("Қолмен PM2.5 орнату")
    ov_cell = st.selectbox("Ұяшық", cells, format_func=fmt, key="ov_cell")
    ov_val = st.number_input("PM2.5, мкг/м³", min_value=0.0, max_value=float(C.MAX_VALID_READING), value=120.0, step=5.0)
    if use_ov:
        overrides[ov_cell] = ov_val

tab_route, tab_cmp, tab_steps, tab_hmm, tab_ml = st.tabs(
    ["Маршрут", "Алгоритмдерді салыстыру", "Қадамдық визуализация", "HMM", "ML болжам"])

# ---------------- есептеу және қате енгізуді өңдеу ----------------
plan, error = None, None
if start == goal:
    error = "Бастау және аяқтау ұяшықтары бірдей. Басқа ұяшық таңдаңыз."
elif start in blocked or goal in blocked:
    error = "Бастау немесе аяқтау ұяшығы тосқауыл етілген."
else:
    try:
        plan = agent.plan(t, horizon, start, goal, alpha, algorithm, blocked, overrides)
    except ValueError as e:
        error = str(e)

with tab_route:
    if error:
        st.error(error)
    elif plan:
        res, perf, ps = plan["result"], plan["perf"], plan["perf_shortest"]
        if not res.found:
            st.warning("Жол табылмады: тосқауылдар бастау мен аяқтауды бөліп тұр.")
            st.plotly_chart(heat_figure(agent.layout, plan["pm"], None, None, start, goal, blocked), width='stretch')
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Қадам саны", perf["steps"], f"{perf['steps'] - ps['steps']:+d} қысқа жолға қарағанда", delta_color="inverse")
            dd = (perf["dose"] - ps["dose"]) / ps["dose"] * 100 if ps["dose"] else 0.0
            c2.metric("Ластану дозасы", f"{perf['dose']:.0f}", f"{dd:+.0f}% қысқа жолға қарағанда", delta_color="inverse")
            c3.metric("Орташа PM2.5", f"{perf['mean_pm']:.0f}")
            c4.metric("Қауіпті ұяшықтар", perf["danger_cells"])
            st.info(agent.explain(plan, alpha))
            st.plotly_chart(heat_figure(agent.layout, plan["pm"], res.path, plan["shortest"].path, start, goal, blocked),
                            width='stretch')
            st.caption("Ромб - датчик, сұр - ғимарат/тосқауыл, үзік сызық - ең қысқа маршрут. "
                       "Доза = маршрут ұяшықтарындағы PM2.5 қосындысы (бастау ұяшығынсыз).")

with tab_cmp:
    if plan:
        rows = []
        for r in agent.compare(plan["pm"], alpha, start, goal, blocked):
            rows.append({"Алгоритм": r.algorithm, "Жол табылды": r.found, "Қадам": r.steps,
                         "Құн": round(r.cost, 3) if r.found else None, "Ашылған түйіндер": r.expanded,
                         "Уақыт, мс": round(r.time_ms, 3)})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width='stretch')
        st.bar_chart(df.set_index("Алгоритм")["Ашылған түйіндер"])
        st.markdown("BFS құнды ескермейді (ең аз қадам), UCS пен A* құн бойынша оңтайлы. "
                    "A* эвристика арқасында түйінді аз ашады. 8×8 тор кішкентай, сондықтан айырмашылық үлкен емес.")
    else:
        st.info("Алдымен дұрыс бастау және аяқтау ұяшығын таңдаңыз.")

with tab_steps:
    if plan and plan["result"].found:
        order = plan["result"].expanded_order
        k = st.slider("Қадам (ашылған түйін)", 1, len(order), len(order))
        st.plotly_chart(heat_figure(agent.layout, plan["pm"], plan["result"].path if k == len(order) else None, None,
                                    start, goal, blocked, explored=order[:k], current=order[k - 1]), width='stretch')
        st.caption(f"{plan['result'].algorithm}: {k}-қадамда ашылған түйін {order[k - 1]}. Сандар - ашылу реті.")
    else:
        st.info("Маршрут табылған кезде қадамдарды көруге болады.")

with tab_hmm:
    S = len(agent.data.sensor_cells)
    si = st.selectbox("Датчик", range(S), format_func=lambda i: f"{i}: ұяшық {agent.data.sensor_cells[i]}")
    lo = max(0, t - 72)
    xs = list(range(lo, t + 1))
    raw = agent.data.readings[lo:t + 1, si]
    est = agent.sensor_est[lo:t + 1, si]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=raw, mode="lines+markers", name="шулы оқылым", line=dict(color="#bbb"), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=xs, y=est, name="HMM бағасы", line=dict(color="#1f4fd8", width=3)))
    if agent.data.pm_true is not None:
        r, c = agent.data.sensor_cells[si]
        fig.add_trace(go.Scatter(x=xs, y=agent.data.pm_true[lo:t + 1, r, c], name="шын мән (симулятор)", line=dict(color="green", dash="dot")))
    for thr in (C.PM_LOW, C.PM_HIGH):
        fig.add_hline(y=thr, line_dash="dash", line_color="#999")
    fig.update_layout(height=380, xaxis_title="сағат индексі", yaxis_title="PM2.5", margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    b = agent.beliefs[t, si]
    st.write("**Қазіргі күй сенімі (Forward):**")
    st.dataframe(pd.DataFrame({"Күй": C.STATES, "Ықтималдық": np.round(b, 3)}), hide_index=True)
    vit = agent.hmm.viterbi(discretize(agent.data.readings[:t + 1, si]))
    st.write(f"**Viterbi (ең ықтимал күйлер тізбегі) соңғы күй:** {C.STATES[int(vit[-1])]}")
    with st.expander("Ауысу матрицасы A"):
        st.dataframe(pd.DataFrame(np.round(agent.hmm.A, 3), index=C.STATES, columns=C.STATES))

with tab_ml:
    hh = st.selectbox("Болжам сағаты", [1, 2, 3], key="ml_h")
    ev = cached_eval(agent, hh, source)
    rows = [{"Модель": k, **{m: round(v, 3) for m, v in vals.items()}} for k, vals in ev.items() if not k.startswith("_")]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(f"Тест: соңғы {ev['_test_size']} уақыт қадамы (хронологиялық бөлу). MAE/RMSE - мкг/м³. "
               "Класс метрикалары бағаланған карта бойынша есептелген.")
    cm = np.array(ev["_confusion_rf"])
    st.write("**Random Forest confusion matrix** (жол - шын класс, баған - болжам):")
    st.dataframe(pd.DataFrame(cm, index=[f"шын: {s}" for s in C.STATES], columns=[f"болжам: {s}" for s in C.STATES]))
    if "_true_acc" in ev:
        ta = ev["_true_acc"]
        st.write("**Шын өріспен салыстыру (тек симулятор):** класс дәлдігі")
        st.dataframe(pd.DataFrame({"Модель": list(ta), "Accuracy": [round(v, 3) for v in ta.values()]}), hide_index=True)
        st.caption("Бұл сан 6 датчиктен 64 ұяшыққа интерполяцияның шектеуін көрсетеді.")
    pm_f = agent.pm_map(t, hh)
    st.write(f"**{hh} сағаттан кейінгі болжам картасы:**")
    st.plotly_chart(heat_figure(agent.layout, pm_f), width='stretch')
