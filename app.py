# OptiBlend — 1×5 Animated Dashboard (ISA‑101) — FINAL
# Auditoría propia: sin caché, claves únicas, autoescala en todos los ejes, sin props inválidas para Plotly
# Un solo archivo listo para Streamlit Cloud

import os
import time
from uuid import uuid4
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ------------------------------
# Configuración base (ISA‑101)
# ------------------------------
st.set_page_config(page_title="OptiBlend — 1×5 Animado (ISA‑101)", layout="wide", page_icon="⚙️")
pio.templates.default = "simple_white"

# Fondo blanco forzado y tipografías legibles
st.markdown(
    """
    <style>
      .stApp, .block-container { background: #FFFFFF !important; }
      .metric { border: 1px solid #eee; border-radius: 10px; padding: 6px; }
      .title-wrap { display:flex; align-items:center; gap:16px; border-bottom:1px solid #E6E6E6; padding:10px 0 14px 0; }
      .t-main { font-weight:800; font-size:26px; color:#111; }
      .t-sub  { color:#555; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Namespace único para keys
if "app_ns" not in st.session_state:
    st.session_state["app_ns"] = f"opb_{uuid4().hex[:8]}"
NS = st.session_state["app_ns"]
key = lambda n: f"{NS}:{n}"

# ------------------------------
# Header con logo fijo desde repo (assets/accenture.png) + título grande
# ------------------------------
col_logo, col_text = st.columns([0.14, 0.86])
with col_logo:
    LOGO_PATH = "assets/accenture.png"
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)
with col_text:
    st.markdown(
        """
        <div class='title-wrap'>
          <div>
            <div class='t-main'>OptiBlend® — 1×5 Animated Dashboard</div>
            <div class='t-sub'>ISA‑101 • Mezcla UGM → P80/−100# / TR → Leyes → Dosificación → Hum_total</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# Controles (sidebar)
# ------------------------------
st.sidebar.header("Controles")
# Polinomio Ácido (kg/t) — coeficientes para rango 5–40
st.sidebar.markdown("**Polinomio Ácido (kg/t):**  ")
st.sidebar.markdown(r"$Acid = a_0 + a_1\,CaCO_3 + a_2\,CuS + a_3\,P80_{mm} + a_4\,Hum_{nat} + a_5\,Finos_{-100\#} + a_6\,(CaCO_3\cdot CuS)$")
a0 = st.sidebar.slider("a0", 0.0, 10.0, 5.0, 0.1, key=key("a0"))
a1 = st.sidebar.slider("a1 (× CaCO3)", 0.0, 3.0, 1.2, 0.1, key=key("a1"))
a2 = st.sidebar.slider("a2 (× CuS)", 0.0, 5.0, 2.0, 0.1, key=key("a2"))
a3 = st.sidebar.slider("a3 (× P80 mm)", 0.0, 1.0, 0.30, 0.01, key=key("a3"))
a4 = st.sidebar.slider("a4 (× HumNat %)", 0.0, 2.0, 0.30, 0.01, key=key("a4"))
a5 = st.sidebar.slider("a5 (× Finos -100# %)", 0.0, 2.0, 0.40, 0.01, key=key("a5"))
a6 = st.sidebar.slider("a6 (× CaCO3·CuS)", 0.0, 1.0, 0.25, 0.01, key=key("a6"))

n_drum = st.sidebar.number_input("Nº Tambores", min_value=1, max_value=4, value=2, step=1, key=key("ndr"))
n_ugm  = st.sidebar.number_input("Nº UGM", min_value=2, max_value=5, value=3, step=1, key=key("nugm"))
hum_obj = st.sidebar.slider("Humedad producto objetivo (%)", 5.0, 15.0, 10.0, 0.1, key=key("humobj"))

# Animación
speed_ms = st.sidebar.slider("Velocidad (ms/frame)", 30, 400, 120, 10, key=key("speed"))
seed = st.sidebar.number_input("Seed", 0, 9999, 42, 1, key=key("seed"))

# ------------------------------
# Simulación robusta (rangos solicitados)
# ------------------------------
np.random.seed(int(seed))
periods = 96  # 2 días @ 30 min
idx = pd.date_range(pd.Timestamp.now().floor("30min") - pd.Timedelta(minutes=30*(periods-1)), periods=periods, freq="30min")

# Mezcla UGM — Dirichlet (proporciones suman 1)
alpha = np.ones(int(n_ugm)) + 0.5
if int(n_ugm) == 3:
    alpha = np.array([2.0, 3.0, 1.6])
mix = np.random.dirichlet(alpha, size=periods)  # (periods, n_ugm)

# Propiedades base por UGM (dentro de rangos)
ugm_props = []
for j in range(int(n_ugm)):
    CuT0 = np.random.uniform(0.2, 1.0)     # %
    CuS0 = np.random.uniform(0.2, 0.8)     # %
    CaC0 = np.random.uniform(1.0, 10.0)    # %
    P80m = np.random.uniform(10.0, 15.0)   # mm
    F100 = np.random.uniform(5.0, 20.0)    # %
    Hnat = np.random.uniform(1.0, 3.0)     # %
    ugm_props.append(dict(CuT=CuT0, CuS=CuS0, CaCO3=CaC0, P80mm=P80m, Finos=F100, HumNat=Hnat))

# Mezcla ponderada
props_arr = np.array([[p["CuT"], p["CuS"], p["CaCO3"], p["P80mm"], p["Finos"], p["HumNat"]] for p in ugm_props])
blend = mix @ props_arr  # (periods, 6)
CuT_blend, CuS_blend, CaCO3_blend, P80_blendmm, Finos_blend, HumN_blend = blend.T

# Tiempo de residencia (TR, min) — soft sensor
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

# Polinomio Ácido (kg/t) — parámetros de sidebar
Acid = a0 + a1*CaCO3_blend + a2*CuS_blend + a3*P80_effmm + a4*HumN_blend + a5*Finos_eff + a6*(CaCO3_blend*CuS_blend)
Acid = np.clip(Acid, 5, 40)

# Refino y Agua (kg/t) + humedad total simplificada
Refino = np.clip(2.0 + 6.0*CuT_blend + 2.5*CuS_blend + 0.2*TR + 0.6*np.maximum(hum_obj - HumN_blend, 0), 0, 40)
agua_req_kgpt = hum_obj*10  # ≈ kg/t por % objetivo
Agua = np.clip(np.maximum(agua_req_kgpt - Refino*0.95 - HumN_blend*10, 0), 0, 40)
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
# Figura 1×5 (autoescala)
# ------------------------------
fig = make_subplots(
    rows=1, cols=5, shared_xaxes=False,
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],
    horizontal_spacing=0.04
)

# Paleta acentos (sobria)
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

# (1) Mezcla UGM (%) — stackgroup (leyenda solo aquí)
for j in range(int(n_ugm)):
    fig.add_trace(
        go.Scatter(x=idx[:cur+1], y=100*mix[:cur+1, j], mode="lines",
                   name=f"UGM{j+1}", stackgroup="ugm", line=dict(width=1.5, color=COL["ugm"][j % len(COL["ugm")] ])),
        row=1, col=1
    )
fig.update_yaxes(title_text="Mezcla UGM (%)", row=1, col=1, autorange=True)

# (2) Tonelaje por tambor (tph) — sin leyenda para evitar solape
for di, (dname, series) in enumerate(feeds.items()):
    fig.add_trace(go.Scatter(x=idx[:cur+1], y=series[:cur+1], mode="lines",
                             name=dname, line=dict(width=1.8, color=COL["tph"][di % len(COL["tph")] ]),
                             showlegend=False), row=1, col=2)
fig.update_yaxes(title_text="tph", row=1, col=2, autorange=True)

# (3) P80 (mm) & −100# (%) y TR (min) en eje derecho — sin leyenda (excepto P80)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=P80_effmm[:cur+1], name="P80 (mm)", mode="lines",
                         line=dict(width=2, color=COL["p80"]), showlegend=True), row=1, col=3, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Finos_eff[:cur+1], name="-100# (%)", mode="lines",
                         line=dict(width=1.5, color=COL["f100"]), showlegend=False), row=1, col=3, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=TR[:cur+1], name="TR (min)", mode="lines",
                         line=dict(width=1, dash="dot", color=COL["tr"]), showlegend=False), row=1, col=3, secondary_y=True)
fig.update_yaxes(title_text="P80 (mm)", row=1, col=3, secondary_y=False, autorange=True)
fig.update_yaxes(title_text="% / min", row=1, col=3, secondary_y=True, autorange=True)

# (4) Leyes (%): todas al eje derecho — sin leyenda (evitar saturación)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=100*CuT_blend[:cur+1], name="CuT %", mode="lines",
                         line=dict(width=1.6), showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=100*CuS_blend[:cur+1], name="CuS %", mode="lines",
                         line=dict(width=1.2, dash="dot"), showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=CaCO3_blend[:cur+1], name="CaCO3 %", mode="lines",
                         line=dict(width=1.2, dash="dash"), showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=HumN_blend[:cur+1], name="HumNat %", mode="lines",
                         line=dict(width=1.2), showlegend=False), row=1, col=4, secondary_y=True)
fig.update_yaxes(title_text="%", row=1, col=4, secondary_y=True, autorange=True)
fig.update_yaxes(title_text="Leyes", row=1, col=4, secondary_y=False, showgrid=False)

# (5) Dosificación (kg/t) + Hum_total (%) — leyenda aquí
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Acid[:cur+1], name="Ácido (kg/t)", mode="lines",
                         line=dict(width=2, color=COL["acid"])), row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Agua[:cur+1], name="Agua (kg/t)", mode="lines",
                         line=dict(width=1.8, color=COL["agua"])), row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Refino[:cur+1], name="Refino (kg/t)", mode="lines",
                         line=dict(width=1.8, color=COL["refino"])), row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Hum_out[:cur+1], name="Hum_total %", mode="lines",
                         line=dict(width=1.6, dash="dot", color=COL["hum"])), row=1, col=5, secondary_y=True)
fig.update_yaxes(title_text="kg/t", row=1, col=5, secondary_y=False, autorange=True)
fig.update_yaxes(title_text="%", row=1, col=5, secondary_y=True, autorange=True)

# Layout general — leyenda compacta arriba, sin solapes
fig.update_layout(
    height=520,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.18, x=1.0, xanchor="right", font=dict(size=10)),
    margin=dict(l=30, r=10, t=50, b=20),
    title_text="(1) Mezcla • (2) TPH • (3) P80 / −100# / TR • (4) Leyes • (5) Dosificación / Hum_total",
)

# Rejillas suaves (ISA‑101)
for c in range(1, 6):
    fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False, row=1, col=c)
    fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False, row=1, col=c)

st.plotly_chart(fig, use_container_width=True, key=key("plot"))

# ------------------------------
# KPIs (frame actual) — sincronizados
# ------------------------------
rowi = cur
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Ácido (kg/t)", f"{Acid[rowi]:.1f}")
    st.metric("Hum_total (%)", f"{Hum_out[rowi]:.1f}")
with k2:
    st.metric("Refino (kg/t)", f"{Refino[rowi]:.1f}")
    st.metric("Agua (kg/t)", f"{Agua[rowi]:.1f}")
with k3:
    st.metric("P80 (mm)", f"{P80_effmm[rowi]:.2f}")
    st.metric("−100# (%)", f"{Finos_eff[rowi]:.1f}")
with k4:
    st.metric("CuT (%)", f"{100*CuT_blend[rowi]:.2f}")
    st.metric("CuS (%)", f"{100*CuS_blend[rowi]:.2f}")
with k5:
    st.metric("CaCO3 (%)", f"{CaCO3_blend[rowi]:.1f}")
    st.metric("HumNat (%)", f"{HumN_blend[rowi]:.2f}")

# Coeficientes visibles bajo la ecuación (auditable)
st.markdown(
    f"""
    **Coeficientes actuales:**  
    \( a_0={a0:.2f}\,,\ a_1={a1:.2f}\,,\ a_2={a2:.2f}\,,\ a_3={a3:.3f}\,,\ a_4={a4:.3f}\,,\ a_5={a5:.3f}\,,\ a_6={a6:.3f} \)
    """
)

# ------------------------------
# Bucle de animación (auditable)
# ------------------------------
if st.session_state[key("playing")]:
    st.session_state[key("i")] = (cur + 1) % periods
    time.sleep(st.session_state[key("speed")] / 1000.0)
    st.rerun()
