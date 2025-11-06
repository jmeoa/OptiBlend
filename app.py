# OptiBlend — Primera pestaña (Timeline limpio, sin caché)
# Streamlit + Plotly — listo para Streamlit Cloud
# --------------------------------------------------------------
# - Historia: 2 días, paso 30 min
# - Dos tambores (T1, T2)
# - Dosificación en kg/t: Ácido, Refino, Agua
# - Humedad resultante (%) en eje secundario
# - SIN uso de @st.cache_* para evitar errores de hashing
# - Botón para limpiar caché global de Streamlit Cloud
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="OptiBlend — Timeline", layout="wide", page_icon="⚙️")

# Paleta (Accenture-inspired)
COLORS = {
    "bg": "#1C1C1C",
    "panel": "#2E2E2E",
    "magenta": "#A100FF",
    "blue": "#0072CE",
    "green": "#82FF70",
    "silver": "#E0E0E0",
}
pio.templates.default = "plotly_dark"

# --------------------------------------------------------------
# Utilidades (SIN cache)
# --------------------------------------------------------------

def make_time_index(periods: int = 192, freq: str = "30min") -> pd.DatetimeIndex:
    periods = int(max(1, periods))
    end = pd.Timestamp.now().floor(freq)
    return pd.date_range(end=end, periods=periods, freq=freq)


def simulate_base_signals(periods: int = 192, freq: str = "30min") -> pd.DataFrame:
    """Simula señales base. No usa decoradores de cache ni recibe objetos no hasheables."""
    idx = make_time_index(periods=periods, freq=freq)
    np.random.seed(23)

    origins = ["RajoA", "RajoB", "Stock1", "Stock2"]
    drums = ["T1", "T2"]

    # Cluster discreto por origen
    cluster_states = ["UGM_A", "UGM_B", "UGM_C"]
    cluster_map = {}
    for o in origins:
        run, k = [], 0
        for t in range(len(idx)):
            if t % np.random.randint(2, 7) == 0:
                k = np.random.choice(len(cluster_states))
            run.append(cluster_states[k])
        cluster_map[o] = run

    # Medias por cluster (valores razonables)
    means = {
        "UGM_A": {"CuT": 0.55, "CuS": 0.25, "CaCO3": 2.0, "P80": 8000, "Hum_in": 6.5},
        "UGM_B": {"CuT": 0.35, "CuS": 0.10, "CaCO3": 5.0, "P80": 12000, "Hum_in": 8.0},
        "UGM_C": {"CuT": 0.85, "CuS": 0.45, "CaCO3": 1.0, "P80": 6000, "Hum_in": 5.0},
    }

    # Series de propiedades por origen
    frames = []
    for o in origins:
        clusters = np.array(cluster_map[o])
        noise = {
            "CuT": np.random.normal(0, 0.04, len(idx)),
            "CuS": np.random.normal(0, 0.03, len(idx)),
            "CaCO3": np.random.normal(0, 0.6, len(idx)),
            "P80": np.random.normal(0, 600, len(idx)),
            "Hum_in": np.random.normal(0, 0.6, len(idx)),
        }
        df_o = pd.DataFrame(index=idx)
        for v in ["CuT", "CuS", "CaCO3", "P80", "Hum_in"]:
            base = np.array([means[c][v] for c in clusters], dtype=float)
            df_o[v] = np.clip(base + noise[v], a_min=0, a_max=None)
        df_o["origin"], df_o["cluster"] = o, clusters
        frames.append(df_o.reset_index().rename(columns={"index": "timestamp"}))
    props = pd.concat(frames, ignore_index=True)

    # Flujos (t/h) estilo OU por tambor
    def ou_series(mu, sigma, theta=0.25, x0=None):
        x = np.zeros(len(idx))
        x[0] = mu if x0 is None else x0
        for t in range(1, len(idx)):
            x[t] = x[t-1] + theta*(mu - x[t-1]) + np.random.normal(0, sigma)
        return np.clip(x, 0.0, None)

    tph_T1, tph_T2 = ou_series(800, 25), ou_series(780, 30, x0=760)

    # Mezcla suave por tambor (proporciones de origen)
    def smooth_dirichlet(k=4, scale=40):
        raw = pd.DataFrame(np.random.dirichlet(np.ones(k), len(idx))).rolling(scale, min_periods=1).mean().values
        raw = raw / raw.sum(axis=1, keepdims=True)
        return raw

    mix_T1, mix_T2 = smooth_dirichlet(), smooth_dirichlet()

    origin_list, drum_list = origins, drums
    blends = []
    for d, tph, M in zip(drum_list, [tph_T1, tph_T2], [mix_T1, mix_T2]):
        df = pd.DataFrame({
            "timestamp": np.tile(idx, len(origin_list)),
            "origin": np.repeat(origin_list, len(idx)),
            "drum": d,
            "tph": np.concatenate([tph * M[:, i] for i in range(len(origin_list))])
        })
        blends.append(df)
    flows = pd.concat(blends, ignore_index=True)

    return flows.merge(props, on=["timestamp", "origin"], how="left")


def weighted_feed(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["timestamp", "drum"]).apply(lambda g: pd.Series({
        "tph": g["tph"].sum(),
        "CuT": np.average(g["CuT"], weights=g["tph"] + 1e-6),
        "CuS": np.average(g["CuS"], weights=g["tph"] + 1e-6),
        "CaCO3": np.average(g["CaCO3"], weights=g["tph"] + 1e-6),
        "P80": np.average(g["P80"], weights=g["tph"] + 1e-6),
        "Hum_in": np.average(g["Hum_in"], weights=g["tph"] + 1e-6),
    })).reset_index()
    return agg

# Modelos de dosificación (kg/t)

def dosing_models(feed: pd.DataFrame, hum_target: float, rho_refino: float) -> pd.DataFrame:
    f = feed.copy()
    acid = (2.0 + 1.4*f["CaCO3"] + 0.8*f["CuS"] + 0.00006*f["P80"] + 0.05*f["Hum_in"] + 0.18*f["CaCO3"]*f["CuS"]).clip(0, 30)

    refino_lpt = (0.8 + 4.0*f["CuT"] + 2.5*f["CuS"] + 35000.0/np.maximum(f["P80"], 2000) + 0.2*np.maximum(hum_target - f["Hum_in"], 0))
    refino_kgpt = refino_lpt * rho_refino

    ms = (100 - f["Hum_in"]) / 100.0
    mw_in = (f["Hum_in"]) / 100.0
    mw_ref_t = refino_kgpt / 1000.0

    def water_needed(ms_row, mw_in_row, mw_add_ref_row):
        target_total_w = (hum_target/100.0) * (ms_row + mw_in_row)
        extra = target_total_w - (mw_in_row + mw_add_ref_row)
        return max(extra, 0.0)

    agua_kgpt = np.array([water_needed(ms_i, mw_i, mwr) for ms_i, mw_i, mwr in zip(ms, mw_in, mw_ref_t)]) * 1000.0

    out = f.copy()
    out["set_Acid_kgpt"], out["set_Refino_kgpt"], out["set_Agua_kgpt"] = acid, refino_kgpt, agua_kgpt
    out["ms"], out["mw_in_t"], out["mw_ref_t"], out["mw_agua_t"] = ms, mw_in, mw_ref_t, agua_kgpt/1000.0
    return out


def humidity_out(df: pd.DataFrame, k_evap: float) -> pd.Series:
    m_loss = k_evap * (df["mw_ref_t"] + df["mw_agua_t"])  # t
    mw_out = df["mw_in_t"] + df["mw_ref_t"] + df["mw_agua_t"] - m_loss
    return (100.0 * mw_out / (df["ms"] + mw_out)).clip(0, 30)

# --------------------------------------------------------------
# UI
# --------------------------------------------------------------

st.markdown(
    f"""
    <div style='background:{COLORS['panel']}; padding:10px; border-radius:12px;'>
      <h2 style='color:{COLORS['magenta']}; margin:0;'>OptiBlend® — Timeline</h2>
      <p style='color:{COLORS['silver']}; margin:0;'>Setpoints (kg/t) y Humedad (%) — ventana 2 días, paso 30 min</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: controles globales + botón para limpiar caché
st.sidebar.header("Parámetros globales")
hum_target = st.sidebar.slider("Humedad objetivo (%)", 6.0, 12.0, 8.5, 0.1)
rho_refino = st.sidebar.slider("Densidad refino (kg/L)", 0.98, 1.10, 1.02, 0.01)
k_evap = st.sidebar.slider("Pérdidas por evaporación k_evap", 0.00, 0.15, 0.04, 0.01)

if st.sidebar.button("🔄 Limpiar caché de Streamlit"):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caché limpiada. Por favor, recarga la app.")
    except Exception as e:
        st.warning(f"No se pudo limpiar caché: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Bias de leyes por origen")
origins = ["RajoA", "RajoB", "Stock1", "Stock2"]
bias = {}
for o in origins:
    with st.sidebar.expander(o):
        bias[(o, "CuT")] = st.slider(f"{o} CuT bias (%)", -0.2, 0.2, 0.0, 0.01)
        bias[(o, "CuS")] = st.slider(f"{o} CuS bias (%)", -0.2, 0.2, 0.0, 0.01)
        bias[(o, "CaCO3")] = st.slider(f"{o} CaCO3 bias (%)", -1.0, 1.0, 0.0, 0.05)
        bias[(o, "P80")] = st.slider(f"{o} P80 bias (µm)", -1500, 1500, 0, 50)
        bias[(o, "Hum_in")] = st.slider(f"{o} Hum_in bias (%)", -1.5, 1.5, 0.0, 0.1)

# Simulación base (SIN caché)
idx = make_time_index(periods=192, freq="30min")
df_base = simulate_base_signals(periods=len(idx), freq="30min")

# Aplicar bias
for (o, v), b in bias.items():
    mask = (df_base["origin"] == o)
    if v == "P80":
        df_base.loc[mask, v] = np.clip(df_base.loc[mask, v] + b, 2000, None)
    else:
        df_base.loc[mask, v] = np.clip(df_base.loc[mask, v] * (1 + b), 0, None)

# Feed ponderado → dosificación → humedad
feed = weighted_feed(df_base)
sets = dosing_models(feed, hum_target=hum_target, rho_refino=rho_refino)
sets["Hum_out_pct"] = humidity_out(sets, k_evap=k_evap)

# Timeline por tambor
st.markdown("### Timeline — Dosificación (kg/t) y Humedad (%)")
for drum in ["T1", "T2"]:
    df_d = sets[sets["drum"] == drum].copy()

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Acid_kgpt"],
                             name="Ácido (kg/t)", line=dict(color=COLORS["magenta"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Refino_kgpt"],
                             name="Refino (kg/t)", line=dict(color=COLORS["blue"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Agua_kgpt"],
                             name="Agua (kg/t)", line=dict(color=COLORS["green"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["Hum_out_pct"],
                             name="Humedad_out (%)", line=dict(color=COLORS["silver"], width=2, dash="dot")),
                  row=1, col=1, secondary_y=True)

    fig.update_layout(
        height=460,
        margin=dict(l=40, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        title_text=f"{drum} — Setpoints y Humedad",
    )
    fig.update_yaxes(title_text="kg/t", secondary_y=False)
    fig.update_yaxes(title_text="%", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# Pie
st.markdown(
    f"<div style='color:{COLORS['silver']}; font-size:12px; padding-top:6px;'>"
    f"OptiBlend® — Primera pestaña: timeline de Ácido/Refino/Agua (kg/t) y Humedad (%) · 2 días · step 30 min."
    f"</div>",
    unsafe_allow_html=True,
)
# OptiBlend — Primera pestaña (Timeline limpio, sin caché)
# Streamlit + Plotly — listo para Streamlit Cloud
# --------------------------------------------------------------
# - Historia: 2 días, paso 30 min
# - Dos tambores (T1, T2)
# - Dosificación en kg/t: Ácido, Refino, Agua
# - Humedad resultante (%) en eje secundario
# - SIN uso de @st.cache_* para evitar errores de hashing
# - Botón para limpiar caché global de Streamlit Cloud
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="OptiBlend — Timeline", layout="wide", page_icon="⚙️")

# Paleta (Accenture-inspired)
COLORS = {
    "bg": "#1C1C1C",
    "panel": "#2E2E2E",
    "magenta": "#A100FF",
    "blue": "#0072CE",
    "green": "#82FF70",
    "silver": "#E0E0E0",
}
pio.templates.default = "plotly_dark"

# --------------------------------------------------------------
# Utilidades (SIN cache)
# --------------------------------------------------------------

def make_time_index(periods: int = 192, freq: str = "30min") -> pd.DatetimeIndex:
    periods = int(max(1, periods))
    end = pd.Timestamp.now().floor(freq)
    return pd.date_range(end=end, periods=periods, freq=freq)


def simulate_base_signals(periods: int = 192, freq: str = "30min") -> pd.DataFrame:
    """Simula señales base. No usa decoradores de cache ni recibe objetos no hasheables."""
    idx = make_time_index(periods=periods, freq=freq)
    np.random.seed(23)

    origins = ["RajoA", "RajoB", "Stock1", "Stock2"]
    drums = ["T1", "T2"]

    # Cluster discreto por origen
    cluster_states = ["UGM_A", "UGM_B", "UGM_C"]
    cluster_map = {}
    for o in origins:
        run, k = [], 0
        for t in range(len(idx)):
            if t % np.random.randint(2, 7) == 0:
                k = np.random.choice(len(cluster_states))
            run.append(cluster_states[k])
        cluster_map[o] = run

    # Medias por cluster (valores razonables)
    means = {
        "UGM_A": {"CuT": 0.55, "CuS": 0.25, "CaCO3": 2.0, "P80": 8000, "Hum_in": 6.5},
        "UGM_B": {"CuT": 0.35, "CuS": 0.10, "CaCO3": 5.0, "P80": 12000, "Hum_in": 8.0},
        "UGM_C": {"CuT": 0.85, "CuS": 0.45, "CaCO3": 1.0, "P80": 6000, "Hum_in": 5.0},
    }

    # Series de propiedades por origen
    frames = []
    for o in origins:
        clusters = np.array(cluster_map[o])
        noise = {
            "CuT": np.random.normal(0, 0.04, len(idx)),
            "CuS": np.random.normal(0, 0.03, len(idx)),
            "CaCO3": np.random.normal(0, 0.6, len(idx)),
            "P80": np.random.normal(0, 600, len(idx)),
            "Hum_in": np.random.normal(0, 0.6, len(idx)),
        }
        df_o = pd.DataFrame(index=idx)
        for v in ["CuT", "CuS", "CaCO3", "P80", "Hum_in"]:
            base = np.array([means[c][v] for c in clusters], dtype=float)
            df_o[v] = np.clip(base + noise[v], a_min=0, a_max=None)
        df_o["origin"], df_o["cluster"] = o, clusters
        frames.append(df_o.reset_index().rename(columns={"index": "timestamp"}))
    props = pd.concat(frames, ignore_index=True)

    # Flujos (t/h) estilo OU por tambor
    def ou_series(mu, sigma, theta=0.25, x0=None):
        x = np.zeros(len(idx))
        x[0] = mu if x0 is None else x0
        for t in range(1, len(idx)):
            x[t] = x[t-1] + theta*(mu - x[t-1]) + np.random.normal(0, sigma)
        return np.clip(x, 0.0, None)

    tph_T1, tph_T2 = ou_series(800, 25), ou_series(780, 30, x0=760)

    # Mezcla suave por tambor (proporciones de origen)
    def smooth_dirichlet(k=4, scale=40):
        raw = pd.DataFrame(np.random.dirichlet(np.ones(k), len(idx))).rolling(scale, min_periods=1).mean().values
        raw = raw / raw.sum(axis=1, keepdims=True)
        return raw

    mix_T1, mix_T2 = smooth_dirichlet(), smooth_dirichlet()

    origin_list, drum_list = origins, drums
    blends = []
    for d, tph, M in zip(drum_list, [tph_T1, tph_T2], [mix_T1, mix_T2]):
        df = pd.DataFrame({
            "timestamp": np.tile(idx, len(origin_list)),
            "origin": np.repeat(origin_list, len(idx)),
            "drum": d,
            "tph": np.concatenate([tph * M[:, i] for i in range(len(origin_list))])
        })
        blends.append(df)
    flows = pd.concat(blends, ignore_index=True)

    return flows.merge(props, on=["timestamp", "origin"], how="left")


def weighted_feed(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["timestamp", "drum"]).apply(lambda g: pd.Series({
        "tph": g["tph"].sum(),
        "CuT": np.average(g["CuT"], weights=g["tph"] + 1e-6),
        "CuS": np.average(g["CuS"], weights=g["tph"] + 1e-6),
        "CaCO3": np.average(g["CaCO3"], weights=g["tph"] + 1e-6),
        "P80": np.average(g["P80"], weights=g["tph"] + 1e-6),
        "Hum_in": np.average(g["Hum_in"], weights=g["tph"] + 1e-6),
    })).reset_index()
    return agg

# Modelos de dosificación (kg/t)

def dosing_models(feed: pd.DataFrame, hum_target: float, rho_refino: float) -> pd.DataFrame:
    f = feed.copy()
    acid = (2.0 + 1.4*f["CaCO3"] + 0.8*f["CuS"] + 0.00006*f["P80"] + 0.05*f["Hum_in"] + 0.18*f["CaCO3"]*f["CuS"]).clip(0, 30)

    refino_lpt = (0.8 + 4.0*f["CuT"] + 2.5*f["CuS"] + 35000.0/np.maximum(f["P80"], 2000) + 0.2*np.maximum(hum_target - f["Hum_in"], 0))
    refino_kgpt = refino_lpt * rho_refino

    ms = (100 - f["Hum_in"]) / 100.0
    mw_in = (f["Hum_in"]) / 100.0
    mw_ref_t = refino_kgpt / 1000.0

    def water_needed(ms_row, mw_in_row, mw_add_ref_row):
        target_total_w = (hum_target/100.0) * (ms_row + mw_in_row)
        extra = target_total_w - (mw_in_row + mw_add_ref_row)
        return max(extra, 0.0)

    agua_kgpt = np.array([water_needed(ms_i, mw_i, mwr) for ms_i, mw_i, mwr in zip(ms, mw_in, mw_ref_t)]) * 1000.0

    out = f.copy()
    out["set_Acid_kgpt"], out["set_Refino_kgpt"], out["set_Agua_kgpt"] = acid, refino_kgpt, agua_kgpt
    out["ms"], out["mw_in_t"], out["mw_ref_t"], out["mw_agua_t"] = ms, mw_in, mw_ref_t, agua_kgpt/1000.0
    return out


def humidity_out(df: pd.DataFrame, k_evap: float) -> pd.Series:
    m_loss = k_evap * (df["mw_ref_t"] + df["mw_agua_t"])  # t
    mw_out = df["mw_in_t"] + df["mw_ref_t"] + df["mw_agua_t"] - m_loss
    return (100.0 * mw_out / (df["ms"] + mw_out)).clip(0, 30)

# --------------------------------------------------------------
# UI
# --------------------------------------------------------------

st.markdown(
    f"""
    <div style='background:{COLORS['panel']}; padding:10px; border-radius:12px;'>
      <h2 style='color:{COLORS['magenta']}; margin:0;'>OptiBlend® — Timeline</h2>
      <p style='color:{COLORS['silver']}; margin:0;'>Setpoints (kg/t) y Humedad (%) — ventana 2 días, paso 30 min</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: controles globales + botón para limpiar caché
st.sidebar.header("Parámetros globales")
hum_target = st.sidebar.slider("Humedad objetivo (%)", 6.0, 12.0, 8.5, 0.1)
rho_refino = st.sidebar.slider("Densidad refino (kg/L)", 0.98, 1.10, 1.02, 0.01)
k_evap = st.sidebar.slider("Pérdidas por evaporación k_evap", 0.00, 0.15, 0.04, 0.01)

if st.sidebar.button("🔄 Limpiar caché de Streamlit"):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caché limpiada. Por favor, recarga la app.")
    except Exception as e:
        st.warning(f"No se pudo limpiar caché: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Bias de leyes por origen")
origins = ["RajoA", "RajoB", "Stock1", "Stock2"]
bias = {}
for o in origins:
    with st.sidebar.expander(o):
        bias[(o, "CuT")] = st.slider(f"{o} CuT bias (%)", -0.2, 0.2, 0.0, 0.01)
        bias[(o, "CuS")] = st.slider(f"{o} CuS bias (%)", -0.2, 0.2, 0.0, 0.01)
        bias[(o, "CaCO3")] = st.slider(f"{o} CaCO3 bias (%)", -1.0, 1.0, 0.0, 0.05)
        bias[(o, "P80")] = st.slider(f"{o} P80 bias (µm)", -1500, 1500, 0, 50)
        bias[(o, "Hum_in")] = st.slider(f"{o} Hum_in bias (%)", -1.5, 1.5, 0.0, 0.1)

# Simulación base (SIN caché)
idx = make_time_index(periods=192, freq="30min")
df_base = simulate_base_signals(periods=len(idx), freq="30min")

# Aplicar bias
for (o, v), b in bias.items():
    mask = (df_base["origin"] == o)
    if v == "P80":
        df_base.loc[mask, v] = np.clip(df_base.loc[mask, v] + b, 2000, None)
    else:
        df_base.loc[mask, v] = np.clip(df_base.loc[mask, v] * (1 + b), 0, None)

# Feed ponderado → dosificación → humedad
feed = weighted_feed(df_base)
sets = dosing_models(feed, hum_target=hum_target, rho_refino=rho_refino)
sets["Hum_out_pct"] = humidity_out(sets, k_evap=k_evap)

# Timeline por tambor
st.markdown("### Timeline — Dosificación (kg/t) y Humedad (%)")
for drum in ["T1", "T2"]:
    df_d = sets[sets["drum"] == drum].copy()

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Acid_kgpt"],
                             name="Ácido (kg/t)", line=dict(color=COLORS["magenta"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Refino_kgpt"],
                             name="Refino (kg/t)", line=dict(color=COLORS["blue"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["set_Agua_kgpt"],
                             name="Agua (kg/t)", line=dict(color=COLORS["green"], width=2)),
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_d["timestamp"], y=df_d["Hum_out_pct"],
                             name="Humedad_out (%)", line=dict(color=COLORS["silver"], width=2, dash="dot")),
                  row=1, col=1, secondary_y=True)

    fig.update_layout(
        height=460,
        margin=dict(l=40, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["panel"],
        title_text=f"{drum} — Setpoints y Humedad",
    )
    fig.update_yaxes(title_text="kg/t", secondary_y=False)
    fig.update_yaxes(title_text="%", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

# Pie
st.markdown(
    f"<div style='color:{COLORS['silver']}; font-size:12px; padding-top:6px;'>"
    f"OptiBlend® — Primera pestaña: timeline de Ácido/Refino/Agua (kg/t) y Humedad (%) · 2 días · step 30 min."
    f"</div>",
    unsafe_allow_html=True,
)
