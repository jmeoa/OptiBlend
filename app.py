# OptiBlend — Subplot 1x5 Animado (ISA‑101)
# Un solo archivo (app.py) listo para Streamlit Cloud
# - 1 fila x 5 columnas en Plotly
# - Animación simple (Play/Pause + slider de frame)
# - Controles: coeficientes del polinomio de dosificación, # tambores, # UGM, humedad objetivo
# - ISA-101: fondo blanco, grillas suaves, pocos acentos, foco en tendencias y alarmas mínimas

import time
from uuid import uuid4
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ------------------------------
# Configuración visual ISA-101
# ------------------------------
st.set_page_config(page_title="OptiBlend — 1x5 Animado (ISA-101)", layout="wide", page_icon="⚙️")
pio.templates.default = "simple_white"

st.markdown(
    """
    <style>
  .stApp, .block-container { background: #F3F4F7 !important; }
  .metric { border: 1px solid #eee; border-radius: 10px; padding: 6px; }
  .acc-header { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #E0E0E0; padding:8px 0 14px 0; }
  .acc-title { font-weight:800; font-size:28px; color:#111; }
  .acc-sub { color:#555; font-size:13px; }
  .kpi-title { font-size:12px; color:#666; margin-bottom:4px; }
</style>
    """,
    unsafe_allow_html=True,
)

# Namespace único para evitar colisiones de keys
if "app_ns" not in st.session_state:
    st.session_state["app_ns"] = f"opb_1x5_{uuid4().hex[:8]}"
NS = st.session_state["app_ns"]
key = lambda n: f"{NS}:{n}"

# ------------------------------
# Header con logo desde repo (sin uploader) + título
# ------------------------------
col_logo, col_title = st.columns([0.15, 0.85])
with col_logo:
    LOGO_PATH = "assets/accenture.png"
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
with col_title:
    st.markdown(
        """
        <div class='acc-header'>
          <div>
            <div class='acc-title'>OptiBlend® — 1x5 Animated Dashboard</div>
            <div class='acc-sub'>ISA‑101 • Mezcla UGM → P80/−100# / TR → Leyes → Dosificación → Humedad</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# Parámetros (sidebar)
# ------------------------------
st.sidebar.header("Controles")
# 1) Coeficientes polinomio (ajustan Ácido entre 5–40 kg/t)
st.sidebar.markdown("**Polinomio Ácido (kg/t)** = a0 + a1·CaCO3 + a2·CuS + a3·P80_mm + a4·HumNat + a5·Finos_100# + a6·CaCO3·CuS")
a0 = st.sidebar.slider("a0", 0.0, 10.0, 5.0, 0.1, key=key("a0"))
a1 = st.sidebar.slider("a1 (× CaCO3)", 0.0, 3.0, 1.2, 0.1, key=key("a1"))
a2 = st.sidebar.slider("a2 (× CuS)", 0.0, 5.0, 2.0, 0.1, key=key("a2"))
a3 = st.sidebar.slider("a3 (× P80 mm)", 0.0, 1.0, 0.30, 0.01, key=key("a3"))
a4 = st.sidebar.slider("a4 (× HumNat %)", 0.0, 2.0, 0.30, 0.01, key=key("a4"))
a5 = st.sidebar.slider("a5 (× Finos -100# %)", 0.0, 2.0, 0.40, 0.01, key=key("a5"))
a6 = st.sidebar.slider("a6 (× CaCO3·CuS)", 0.0, 1.0, 0.25, 0.01, key=key("a6"))

# 2) Nº tambores y 3) Nº UGM
n_drum = st.sidebar.number_input("Nº Tambores", min_value=1, max_value=4, value=2, step=1, key=key("ndr"))
n_ugm  = st.sidebar.number_input("Nº UGM", min_value=2, max_value=5, value=3, step=1, key=key("nugm"))

# 4) Humedad de producto objetivo
hum_obj = st.sidebar.slider("Humedad producto objetivo (%)", 5.0, 15.0, 10.0, 0.1, key=key("humobj"))

# Animación
speed_ms = st.sidebar.slider("Velocidad (ms/frame)", 30, 400, 120, 10, key=key("speed"))
seed = st.sidebar.number_input("Seed", 0, 9999, 42, 1, key=key("seed"))

# ------------------------------
# Datos simulados
# ------------------------------
np.random.seed(int(seed))
periods = 96  # 2 días @ 30 min
idx = pd.date_range(pd.Timestamp.now().floor("30min") - pd.Timedelta(minutes=30*(periods-1)), periods=periods, freq="30min")

# Mezcla de UGM (proporciones que suman 1) usando Dirichlet
if n_ugm == 3:
    alpha = np.array([2.0, 3.0, 1.5])
else:  # generalizar: pesos uniformes
    alpha = np.ones(int(n_ugm)) + 0.5
mix = np.random.dirichlet(alpha, size=periods)  # shape (periods, n_ugm)

# Definir propiedades base por UGM (diferenciadas) dentro de rangos dados
# RANGOS:
# CuT 0.2–1.0 %; CuS 0.2–0.8 %; CaCO3 1–10 %; P80 10–15 mm; Finos -100# 5–20 %; HumNat 1–3 %
ugm_props = []
for j in range(int(n_ugm)):
    CuT = np.random.uniform(0.25, 0.95)
    CuS = np.random.uniform(0.25, 0.75)
    CaC = np.random.uniform(1.5, 9.0)
    P80mm = np.random.uniform(10.2, 14.5)
    F100 = np.random.uniform(6.0, 19.0)
    Hnat = np.random.uniform(1.2, 2.8)
    ugm_props.append(dict(CuT=CuT, CuS=CuS, CaCO3=CaC, P80mm=P80mm, Finos=F100, HumNat=Hnat))

# Mezclado ponderado por proporción UGM en cada instante
CuT_blend   = np.zeros(periods)
CuS_blend   = np.zeros(periods)
CaCO3_blend = np.zeros(periods)
P80_blendmm = np.zeros(periods)
Finos_blend = np.zeros(periods)
HumN_blend  = np.zeros(periods)
for t in range(periods):
    w = mix[t]
    props = np.array([[p["CuT"], p["CuS"], p["CaCO3"], p["P80mm"], p["Finos"], p["HumNat"]] for p in ugm_props])
    v = (w @ props)  # mezcla lineal
    CuT_blend[t], CuS_blend[t], CaCO3_blend[t], P80_blendmm[t], Finos_blend[t], HumN_blend[t] = v

# Tiempo de residencia estimado por soft-sensor (min)
# Mayor TR → P80 más bajo y Finos más alto (aplicamos una modulación suave)
TR = 8 + 4*np.sin(np.linspace(0, 2*np.pi, periods))  # 8–12 min
P80_effmm = np.clip(P80_blendmm * (1.0 - 0.03*(TR-10)), 10.0, 15.0)
Finos_eff = np.clip(Finos_blend * (1.0 + 0.02*(TR-10)), 5.0, 25.0)

# Tonelaje por tambor (tph)
ndr = int(n_drum)
feeds = {}
for d in range(ndr):
    base = np.random.uniform(800, 1200)
    noise = 120*np.sin(np.linspace(0, 3*np.pi, periods) + d)
    feeds[f"T{d+1}"] = np.clip(base + noise + np.random.normal(0, 30, periods), 500, 1500)

# ------------------------------
# Dosificaciones y humedad producto
# ------------------------------
# Polinomio Ácido (kg/t)
Acid = a0 + a1*CaCO3_blend + a2*CuS_blend + a3*P80_effmm + a4*HumN_blend + a5*Finos_eff + a6*(CaCO3_blend*CuS_blend)
Acid = np.clip(Acid, 5, 40)

# Refino (kg/t) — lineal en CuT, CuS, TR y gap de humedad
Refino = np.clip( 2.0 + 6.0*CuT_blend + 2.5*CuS_blend + 0.2*TR + 0.6*np.maximum(hum_obj - HumN_blend, 0), 0, 40)

# Agua (kg/t) para alcanzar humedad objetivo total del producto (simplificado):
# Suponemos masa seca = 1 t ⇒ kg agua objetivo = hum_obj% de masa húmeda (~aprox)
agua_req_kgpt = hum_obj*10  # 10 kg/t por cada 1% aprox
Agua = np.clip(np.maximum(agua_req_kgpt - Refino*0.95 - HumN_blend*10, 0), 0, 40)  # resta el aporte de refino y hum nat

# Humedad total de soluciones (%) aproximada
Hum_out = np.clip(HumN_blend + 0.095*Refino + 0.10*Agua, 5, 18)

# ------------------------------
# Estado de animación
# ------------------------------
if key("i") not in st.session_state:
    st.session_state[key("i")] = 0
if key("playing") not in st.session_state:
    st.session_state[key("playing")] = False

# Botón único Play/Pause
label = "⏸ Pause" if st.session_state[key("playing")] else "▶ Play"
if st.button(label, key=key("pp")):
    st.session_state[key("playing")] = not st.session_state[key("playing")]

# Slider de frame
st.session_state[key("i")] = st.slider("Frame", 0, periods-1, st.session_state[key("i")], 1, key=key("frame"))
cur = int(st.session_state[key("i")])

# ------------------------------
# Figura 1x5 con autoescalado
# ------------------------------
fig = make_subplots(
    rows=1, cols=5, shared_xaxes=False,
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
    horizontal_spacing=0.04,
    subplot_titles=(
        "Mezcla UGM (%)",
        "Tonelaje por tambor (tph)",
        "P80 / −100# / TR",
        "Leyes (%)",
        "Dosificación (kg/t) y Hum_total (%)"
    )
)

# Paleta sobria (ISA‑101 / acentos discretos)
COL = {
    "ugm": ["#7E57C2", "#D81B60", "#8E24AA", "#1E88E5", "#43A047"],
    "tph": ["#455A64", "#9E9E9E", "#607D8B", "#78909C"],
    "p80": "#F9A825",
    "f100": "#6D4C41",
    "tr": "#616161",
    "acid": "#A100FF",
    "agua": "#2ECC71",
    "refino": "#0072CE",
    "hum": "#2B2B2B",
}

# (1) Mezcla UGM: líneas apiladas (stackgroup)
for j in range(int(n_ugm)):
    fig.add_trace(
        go.Scatter(
            x=idx[:cur+1], y=100*mix[:cur+1, j], mode="lines",
            name=f"UGM{j+1}", stackgroup="ugm",
            line=dict(width=1.6, color=COL["ugm"][j % len(COL["ugm"])])
        ),
        row=1, col=1
    )
fig.update_yaxes(title_text="Mezcla UGM (%)", row=1, col=1, autorange=True)

# (2) Tonelaje por tambor (tph)
for di, (dname, series) in enumerate(feeds.items()):
    fig.add_trace(
        go.Scatter(
            x=idx[:cur+1], y=series[:cur+1], mode="lines",
            name=dname, showlegend=False,
            line=dict(width=1.8, color=COL["tph"][di % len(COL["tph"])])
        ),
        row=1, col=2
    )
fig.update_yaxes(title_text="tph", row=1, col=2, autorange=True)

# (3) P80 (mm) & Finos -100# (%) con TR (min) en seg eje
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=P80_effmm[:cur+1], name="P80 (mm)", mode="lines",
               line=dict(width=2, color=COL["p80"]), showlegend=True),
    row=1, col=3, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Finos_eff[:cur+1], name="-100# (%)", mode="lines",
               line=dict(width=1.5, color=COL["f100"]), showlegend=False),
    row=1, col=3, secondary_y=True
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=TR[:cur+1], name="TR (min)", mode="lines",
               line=dict(width=1, dash="dot", color=COL["tr"]), showlegend=False),
    row=1, col=3, secondary_y=True
)
fig.update_yaxes(title_text="P80 (mm)", row=1, col=3, secondary_y=False, autorange=True)
fig.update_yaxes(title_text="% / min", row=1, col=3, secondary_y=True, autorange=True)

# (4) Leyes: CuT, CuS, CaCO3, HumNat (%) — todas a eje derecho; eje izq vacío pero se etiqueta
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=100*CuT_blend[:cur+1], name="CuT %", mode="lines",
               line=dict(width=1.6, color="#424242"), showlegend=False),
    row=1, col=4, secondary_y=True
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=100*CuS_blend[:cur+1], name="CuS %", mode="lines",
               line=dict(width=1.2, dash="dot", color="#7B1FA2"), showlegend=False),
    row=1, col=4, secondary_y=True
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=CaCO3_blend[:cur+1], name="CaCO3 %", mode="lines",
               line=dict(width=1.2, dash="dash", color="#1565C0"), showlegend=False),
    row=1, col=4, secondary_y=True
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=HumN_blend[:cur+1], name="HumNat %", mode="lines",
               line=dict(width=1.2, color="#00897B"), showlegend=False),
    row=1, col=4, secondary_y=True
)
fig.update_yaxes(title_text="%", row=1, col=4, secondary_y=True, autorange=True)
fig.update_yaxes(title_text="Leyes", row=1, col=4, secondary_y=False, showgrid=False)

# (5) Dosificación: Ácido, Agua, Refino (kg/t) + Hum_out (%) en eje 2
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Acid[:cur+1], name="Ácido (kg/t)", mode="lines",
               line=dict(width=2, color=COL["acid"])),
    row=1, col=5, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Agua[:cur+1], name="Agua (kg/t)", mode="lines",
               line=dict(width=1.8, color=COL["agua"])),
    row=1, col=5, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Refino[:cur+1], name="Refino (kg/t)", mode="lines",
               line=dict(width=1.8, color=COL["refino"])),
    row=1, col=5, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Hum_out[:cur+1], name="Hum_total %", mode="lines",
               line=dict(width=1.6, dash="dot", color=COL["hum"])),
    row=1, col=5, secondary_y=True
)
fig.update_yaxes(title_text="kg/t", row=1, col=5, secondary_y=False, autorange=True)
fig.update_yaxes(title_text="%", row=1, col=5, secondary_y=True, autorange=True)

# Layout general (fondo blanco + leyenda compacta)
fig.update_layout(
    height=540,
    hovermode="x unified",
    paper_bgcolor="#F5F6FA",
    plot_bgcolor="#F7F7F9",
    font=dict(color="#111", size=12),
    legend=dict(orientation="h", y=1.10, x=1.0, xanchor="right", font=dict(size=11), tracegroupgap=8),
    margin=dict(l=30, r=10, t=90, b=30)
),
    legend=dict(orientation="h", y=1.18, x=1.0, xanchor="right", font=dict(size=10)),
    margin=dict(l=30, r=10, t=50, b=20),
    title_text="(1) Mezcla • (2) TPH • (3) P80 / -100# / TR • (4) Leyes • (5) Dosificación / Hum_total",
)

# Rejillas + ejes profesionales (ISA‑101)
for c in range(1, 6):
    fig.update_xaxes(
        showgrid=True, gridcolor="#E1E3E8", zeroline=False,
        showline=True, linewidth=1, linecolor="#BDBDBD", mirror=True,
        ticks="outside", tickwidth=1, tickcolor="#9E9E9E", ticklen=4,
        automargin=True,
        row=1, col=c,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#E1E3E8", zeroline=False,
        showline=True, linewidth=1, linecolor="#BDBDBD", mirror=True,
        ticks="outside", tickwidth=1, tickcolor="#9E9E9E", ticklen=4,
        automargin=True, title_standoff=6,
        row=1, col=c,
    )

st.plotly_chart(fig, use_container_width=True, key=key("plot"))

# ------------------------------
# KPIs (frame actual)
# ------------------------------
row = cur
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Ácido (kg/t)", f"{Acid[row]:.1f}")
    st.metric("Hum_total (%)", f"{Hum_out[row]:.1f}")
with k2:
    st.metric("Refino (kg/t)", f"{Refino[row]:.1f}")
    st.metric("Agua (kg/t)", f"{Agua[row]:.1f}")
with k3:
    st.metric("P80 (mm)", f"{P80_effmm[row]:.2f}")
    st.metric("-100# (%)", f"{Finos_eff[row]:.1f}")
with k4:
    st.metric("CuT (%)", f"{100*CuT_blend[row]:.2f}")
    st.metric("CuS (%)", f"{100*CuS_blend[row]:.2f}")
with k5:
    st.metric("CaCO3 (%)", f"{CaCO3_blend[row]:.1f}")
    st.metric("HumNat (%)", f"{HumN_blend[row]:.2f}")

# Fórmula del polinomio (render)
st.markdown(
    f"""
    **Modelo Ácido (kg/t):**  
    \( Acid = {a0:.2f} + {a1:.2f}\cdot CaCO_3 + {a2:.2f}\cdot CuS + {a3:.3f}\cdot P80_{{mm}} + {a4:.3f}\cdot Hum_{{nat}} + {a5:.3f}\cdot Finos_{{-100\#}} + {a6:.3f}\cdot (CaCO_3\cdot CuS) \)
    """
)

# ------------------------------
# Animación (toggle Play/Pause)
# ------------------------------
if st.session_state[key("playing")]:
    st.session_state[key("i")] = (cur + 1) % periods
    time.sleep(st.session_state[key("speed")]/1000.0)
    st.rerun()
