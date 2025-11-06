# OptiBlend — Mini App Animada (ligera)
# Objetivo: animar el impacto de la mezcla UGM_A/UGM_B sobre la dosificación y humedad
# Estrategia: animación controlada por Streamlit (sin frames de Plotly) para evitar estáticos
# - 2 días, paso 30 min (96 pasos)
# - Un tambor (T1)
# - UGM_A con mayor requerimiento de ácido que UGM_B
# - Recalcula: P80, dosificación (Ácido/Refino/Agua en kg/t) y Hum_out (%)
# - Muestra composición UGM_A/UGM_B como proporción (área apilada)
# --------------------------------------------------------------

import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

st.set_page_config(page_title="OptiBlend — Mini Animación", layout="wide", page_icon="⚙️")
pio.templates.default = "simple_white"

# Header claro ISA‑101 (alto contraste, sin decoraciones innecesarias)
st.markdown(
    """
    <div style='background:#FFFFFF; padding:10px 12px; border-bottom:1px solid #E6E6E6;'>
      <div style='font-size:22px; font-weight:600; color:#111;'>OptiBlend® — Mini Animación</div>
      <div style='font-size:13px; color:#444;'>UGM_A requiere más ácido que UGM_B • Observe impacto en setpoints (kg/t) y humedad</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Paleta breve (Accenture-inspired)
C = {"bg":"#1C1C1C","panel":"#2E2E2E","magenta":"#A100FF","blue":"#0072CE","green":"#82FF70","silver":"#E0E0E0"}

st.markdown(f"""
<div style='background:{C['panel']}; padding:10px; border-radius:12px;'>
  <h2 style='color:{C['magenta']}; margin:0;'>OptiBlend® — Mini Animación</h2>
  <p style='color:{C['silver']}; margin:0;'>UGM_A > UGM_B en requerimiento de ácido. Ver impacto en setpoints (kg/t) y humedad.</p>
</div>
""", unsafe_allow_html=True)

# --- Namespacing para claves únicas por sesión ---
from uuid import uuid4
# Namespace único por instancia, incluso si coexiste con otras páginas
if 'app_ns' not in st.session_state:
    st.session_state['app_ns'] = f"opb_mini_{uuid4().hex[:8]}"
NS = st.session_state['app_ns']

def k(name: str) -> str:
    return f"{NS}:{name}"

# --------------------------------------------------------------
# Controles
# --------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    hum_target = st.slider("Humedad objetivo (%)", 6.0, 12.0, 8.5, 0.1, key=k("hum_target"))
with col2:
    rho_refino = st.slider("ρ refino (kg/L)", 0.98, 1.10, 1.02, 0.01, key=k("rho_refino"))
with col3:
    mix_amp = st.slider("Amplitud cambio mezcla", 0.0, 1.0, 0.6, 0.05, key=k("mix_amp"))
with col4:
    seed = st.number_input("Seed", 0, 9999, 23, 1, key=k("seed"))
with col5:
    speed_ms = st.slider("Velocidad (ms/frame)", 20, 500, 80, 10, key=k("speed"))

# --------------------------------------------------------------
# Simulación (un tambor)
# --------------------------------------------------------------
np.random.seed(int(seed))
periods = 96  # 2 días * 48 pasos/día
freq_minutes = 30
start = pd.Timestamp.now().floor(f"{freq_minutes}min") - pd.Timedelta(minutes=freq_minutes*(periods-1))
idx = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")

# Señal de mezcla (0..1) — share de UGM_B
phi = np.linspace(0, 2*np.pi, periods)
share_B = 0.5 + mix_amp * 0.5 * np.sin(phi)
share_B = np.clip(share_B, 0.0, 1.0)
share_A = 1.0 - share_B
share_B_pct = 100.0 * share_B
share_A_pct = 100.0 * share_A

# Extremos UGM_A vs UGM_B (A más reactiva / mayor consumo)
extA = {"CuT":0.40, "CuS":0.28, "CaCO3":6.0, "P80":9000, "Hum_in":6.5}
extB = {"CuT":0.90, "CuS":0.10, "CaCO3":2.0, "P80":6000, "Hum_in":5.0}

# Interpolación de propiedades "mezcla observable"
interp = lambda a,b,w: a*(1-w) + b*w
CuT = interp(extA["CuT"],  extB["CuT"],  share_B) + np.random.normal(0, 0.03, periods)
CuS = interp(extA["CuS"],  extB["CuS"],  share_B) + np.random.normal(0, 0.02, periods)
CaC = interp(extA["CaCO3"],extB["CaCO3"],share_B) + np.random.normal(0, 0.4, periods)
P80 = interp(extA["P80"],  extB["P80"],  share_B) + np.random.normal(0, 300, periods)
Hin = interp(extA["Hum_in"],extB["Hum_in"],share_B) + np.random.normal(0, 0.3, periods)

# Límites
CuT = np.clip(CuT, 0.05, 2.0)
CuS = np.clip(CuS, 0.0, 1.0)
CaC = np.clip(CaC, 0.0, 10.0)
P80 = np.clip(P80, 2000, 20000)
Hin = np.clip(Hin, 2.0, 14.0)

# --------------------------------------------
# Modelos — Ácido con identidad explícita UGM
# --------------------------------------------
# Calcula requerimiento de ácido por UGM (usando sus extremos) y luego mezcla por proporción
acid_A = (4.0 + 1.8*extA["CaCO3"] + 1.1*extA["CuS"] + 0.00008*extA["P80"] + 0.06*extA["Hum_in"] + 0.24*extA["CaCO3"]*extA["CuS"])  # mayor base
acid_B = (1.6 + 0.9*extB["CaCO3"] + 0.5*extB["CuS"] + 0.00004*extB["P80"] + 0.03*extB["Hum_in"] + 0.12*extB["CaCO3"]*extB["CuS"])  # menor base

# Suaviza con pequeña dependencia a propiedades mezcladas para que responda con P80/Hin
acid_mix_term = 0.00002*P80 + 0.03*CaC + 0.02*CuS + 0.02*Hin
Acid = (share_A*acid_A + share_B*acid_B) + acid_mix_term
Acid = np.clip(Acid, 0, 35)

# Refino (kg/t) — mantiene lógica basada en propiedades mezcladas
Ref_lpt = (0.8 + 4.0*CuT + 2.2*CuS + 35000.0/np.maximum(P80, 2000) + 0.25*np.maximum(hum_target - Hin, 0))
Ref_kgpt = Ref_lpt * rho_refino

# Balance agua-humedad
ms = (100 - Hin)/100.0
mw_in = Hin/100.0
mw_ref = Ref_kgpt/1000.0
extra_w = (hum_target/100.0)*(ms + mw_in) - (mw_in + mw_ref)
extra_w = np.maximum(extra_w, 0.0)
Wat_kgpt = extra_w * 100.0 * 10.0  # = extra_w*1000 (kg/t)
Hum_out = 100.0 * (mw_in + mw_ref + extra_w) / (ms + mw_in + mw_ref + extra_w)

T = pd.DataFrame({
    "timestamp": idx,
    "CuT": CuT,
    "CuS": CuS,
    "CaCO3": CaC,
    "P80": P80,
    "Hum_in": Hin,
    "Acid_kgpt": Acid,
    "Refino_kgpt": Ref_kgpt,
    "Agua_kgpt": Wat_kgpt,
    "Hum_out_pct": Hum_out,
    "%UGM_A": share_A_pct,
    "%UGM_B": share_B_pct,
})

# --------------------------------------------------------------
# Estado de animación en sesión
# --------------------------------------------------------------
if k("i") not in st.session_state:
    st.session_state[k("i")] = 0
if k("playing") not in st.session_state:
    st.session_state[k("playing")] = False

# Controles de reproducción
# Barra de control (alineada a la izquierda, compacta)
btn_l, btn_m, btn_r = st.columns([0.12, 0.12, 0.76])
with btn_l:
    if st.button("▶ Play", key=k("play")):
        st.session_state[k("playing")] = True
with btn_m:
    if st.button("⏸ Pause", key=k("pause")):
        st.session_state[k("playing")] = False

# Slider manual de frame (sincronizado)
st.session_state[k("i")] = st.slider("Frame", 0, len(T)-1, st.session_state[k("i")], 1, key=k("frame"))
cur = st.session_state[k("i")]

# --------------------------------------------------------------
# Figura (3 filas):
#  Fila 1 → setpoints (kg/t) + Hum_out (%)
#  Fila 2 → composición UGM_A/UGM_B (área apilada 0..100%)
#  Fila 3 → propiedades mezcladas: P80 (µm, eje izq) + leyes/prop (%: CuT, CuS, CaCO3, Hum_in) en eje der
# --------------------------------------------------------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    specs=[[{"secondary_y": True}],
           [{"secondary_y": False}],
           [{"secondary_y": True}]],
    vertical_spacing=0.08
)

# Fila 1 — setpoints + humedad
fig.add_trace(go.Scatter(name="Ácido (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Acid_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["magenta"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Refino (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Refino_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["blue"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Agua (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Agua_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["green"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Humedad_out (%)", x=T["timestamp"].iloc[:cur+1], y=T["Hum_out_pct"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["silver"], width=2, dash="dot")), row=1, col=1, secondary_y=True)

# Fila 2 — composición (área apilada)
# Fila 2 — composición (barras apiladas 0..100%)
fig.add_trace(go.Bar(name="% UGM_A", x=T["timestamp"].iloc[:cur+1], y=T["%UGM_A"].iloc[:cur+1], marker_color="#9B59B6"), row=2, col=1)
fig.add_trace(go.Bar(name="% UGM_B", x=T["timestamp"].iloc[:cur+1], y=T["%UGM_B"].iloc[:cur+1], marker_color="#E91E63"), row=2, col=1)
fig.update_layout(barmode="stack")

# Fila 3 — propiedades
fig.add_trace(go.Scatter(name="P80 (µm)", x=T["timestamp"].iloc[:cur+1], y=T["P80"].iloc[:cur+1],
                         mode="lines", line=dict(color="#FBC02D", width=2)), row=3, col=1, secondary_y=False)
# Eje derecho: leyes/prop (%)
fig.add_trace(go.Scatter(name="CuT (%)", x=T["timestamp"].iloc[:cur+1], y=(T["CuT"].iloc[:cur+1]*100.0),
                         mode="lines", line=dict(width=1)), row=3, col=1, secondary_y=True)
fig.add_trace(go.Scatter(name="CuS (%)", x=T["timestamp"].iloc[:cur+1], y=(T["CuS"].iloc[:cur+1]*100.0),
                         mode="lines", line=dict(width=1, dash="dot")), row=3, col=1, secondary_y=True)
fig.add_trace(go.Scatter(name="CaCO3 (%)", x=T["timestamp"].iloc[:cur+1], y=T["CaCO3"].iloc[:cur+1],
                         mode="lines", line=dict(width=1, dash="dash")), row=3, col=1, secondary_y=True)
fig.add_trace(go.Scatter(name="Hum_in (%)", x=T["timestamp"].iloc[:cur+1], y=T["Hum_in"].iloc[:cur+1],
                         mode="lines", line=dict(width=1)), row=3, col=1, secondary_y=True)

fig.update_layout(
    height=880,
    hovermode="x unified",
    title_text="T1 — Setpoints (kg/t), Humedad (%) y Composición/Propiedades",
    margin=dict(l=40, r=20, t=50, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0, font=dict(size=11), itemwidth=40),
)
# Rejillas suaves y etiquetas legibles (ISA‑101)
for r in (1,2,3):
    fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False, row=r, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False, row=r, col=1)
fig.update_yaxes(title_text="kg/t", secondary_y=False, row=1, col=1)
fig.update_yaxes(title_text="%", secondary_y=True, row=1, col=1)
fig.update_yaxes(title_text="Composición UGM (%)", row=2, col=1, range=[0,100])
fig.update_yaxes(title_text="P80 (µm)", secondary_y=False, row=3, col=1)
fig.update_yaxes(title_text="Leyes / Propiedades (%)", secondary_y=True, row=3, col=1)

# Marcadores del último punto (mejor visibilidad ISA‑101)
lastx = T["timestamp"].iloc[cur]
fig.add_trace(go.Scatter(x=[lastx], y=[T["Acid_kgpt"].iloc[cur]], mode="markers", marker=dict(size=9, color="#A100FF"), name="•"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=[lastx], y=[T["Refino_kgpt"].iloc[cur]], mode="markers", marker=dict(size=9, color="#0072CE"), name="•"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=[lastx], y=[T["Agua_kgpt"].iloc[cur]], mode="markers", marker=dict(size=9, color="#2ECC71"), name="•"), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=[lastx], y=[T["Hum_out_pct"].iloc[cur]], mode="markers", marker=dict(size=9, color="#333333"), name="•"), row=1, col=1, secondary_y=True)

st.plotly_chart(fig, use_container_width=True, key=k("plot"))

# Panel de KPIs
st.markdown("### Estado instantáneo (frame actual)")
row = T.iloc[cur]
K1, K2, K3, K4 = st.columns(4)
with K1:
    st.metric("% UGM_A", f"{row['%UGM_A']:.1f}%")
    st.metric("% UGM_B", f"{row['%UGM_B']:.1f}%")
with K2:
    st.metric("P80 (µm)", f"{row['P80']:.0f}")
    st.metric("Hum_in (%)", f"{row['Hum_in']:.2f}")
with K3:
    st.metric("Ácido (kg/t)", f"{row['Acid_kgpt']:.2f}")
    st.metric("Refino (kg/t)", f"{row['Refino_kgpt']:.2f}")
with K4:
    st.metric("Agua (kg/t)", f"{row['Agua_kgpt']:.2f}")
    st.metric("Humedad_out (%)", f"{row['Hum_out_pct']:.2f}")

# Bucle de reproducción (incrementa frame y rerun)
if st.session_state[k("playing")]:
    nxt = (cur + 1) % len(T)
    st.session_state[k("i")] = nxt
    time.sleep(speed_ms / 1000.0)
    st.experimental_rerun()

# Pie
st.markdown(
    f"<div style='color:{C['silver']}; font-size:12px; padding-top:6px;'>"
    f"OptiBlend® — Mini animación (96 pasos, 2 días, step 30 min). UGM_A demanda más ácido que UGM_B."
    f"</div>",
    unsafe_allow_html=True,
)
# OptiBlend — Mini App Animada (ligera)
# Objetivo: animar el impacto de la mezcla UGM_A/UGM_B sobre la dosificación y humedad
# Estrategia: animación controlada por Streamlit (sin frames de Plotly) para evitar estáticos
# - 2 días, paso 30 min (96 pasos)
# - Un tambor (T1)
# - UGM_A con mayor requerimiento de ácido que UGM_B
# - Recalcula: P80, dosificación (Ácido/Refino/Agua en kg/t) y Hum_out (%)
# - Muestra composición UGM_A/UGM_B como proporción (área apilada)
# --------------------------------------------------------------

import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

st.set_page_config(page_title="OptiBlend — Mini Animación", layout="wide", page_icon="⚙️")
pio.templates.default = "plotly_dark"

# Paleta breve (Accenture-inspired)
C = {"bg":"#1C1C1C","panel":"#2E2E2E","magenta":"#A100FF","blue":"#0072CE","green":"#82FF70","silver":"#E0E0E0"}

st.markdown(f"""
<div style='background:{C['panel']}; padding:10px; border-radius:12px;'>
  <h2 style='color:{C['magenta']}; margin:0;'>OptiBlend® — Mini Animación</h2>
  <p style='color:{C['silver']}; margin:0;'>UGM_A > UGM_B en requerimiento de ácido. Ver impacto en setpoints (kg/t) y humedad.</p>
</div>
""", unsafe_allow_html=True)

# --- Namespacing para claves únicas por sesión ---
if 'run_id' not in st.session_state:
    st.session_state['run_id'] = f"{np.random.randint(0,1_000_000)}"
RUN = st.session_state['run_id']

def k(name: str) -> str:
    return f"mini_{name}_{RUN}"

# --------------------------------------------------------------
# Controles
# --------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
with col1:
    hum_target = st.slider("Humedad objetivo (%)", 6.0, 12.0, 8.5, 0.1, key=k("hum_target"))
with col2:
    rho_refino = st.slider("ρ refino (kg/L)", 0.98, 1.10, 1.02, 0.01, key=k("rho_refino"))
with col3:
    mix_amp = st.slider("Amplitud cambio mezcla", 0.0, 1.0, 0.6, 0.05, key=k("mix_amp"))
with col4:
    seed = st.number_input("Seed", 0, 9999, 23, 1, key=k("seed"))
with col5:
    speed_ms = st.slider("Velocidad (ms/frame)", 20, 500, 80, 10, key=k("speed"))

# --------------------------------------------------------------
# Simulación (un tambor)
# --------------------------------------------------------------
np.random.seed(int(seed))
periods = 96  # 2 días * 48 pasos/día
freq_minutes = 30
start = pd.Timestamp.now().floor(f"{freq_minutes}min") - pd.Timedelta(minutes=freq_minutes*(periods-1))
idx = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")

# Señal de mezcla (0..1) — share de UGM_B
phi = np.linspace(0, 2*np.pi, periods)
share_B = 0.5 + mix_amp * 0.5 * np.sin(phi)
share_B = np.clip(share_B, 0.0, 1.0)
share_A = 1.0 - share_B
share_B_pct = 100.0 * share_B
share_A_pct = 100.0 * share_A

# Extremos UGM_A vs UGM_B (A más reactiva / mayor consumo)
extA = {"CuT":0.40, "CuS":0.28, "CaCO3":6.0, "P80":9000, "Hum_in":6.5}
extB = {"CuT":0.90, "CuS":0.10, "CaCO3":2.0, "P80":6000, "Hum_in":5.0}

# Interpolación de propiedades "mezcla observable"
interp = lambda a,b,w: a*(1-w) + b*w
CuT = interp(extA["CuT"],  extB["CuT"],  share_B) + np.random.normal(0, 0.03, periods)
CuS = interp(extA["CuS"],  extB["CuS"],  share_B) + np.random.normal(0, 0.02, periods)
CaC = interp(extA["CaCO3"],extB["CaCO3"],share_B) + np.random.normal(0, 0.4, periods)
P80 = interp(extA["P80"],  extB["P80"],  share_B) + np.random.normal(0, 300, periods)
Hin = interp(extA["Hum_in"],extB["Hum_in"],share_B) + np.random.normal(0, 0.3, periods)

# Límites
CuT = np.clip(CuT, 0.05, 2.0)
CuS = np.clip(CuS, 0.0, 1.0)
CaC = np.clip(CaC, 0.0, 10.0)
P80 = np.clip(P80, 2000, 20000)
Hin = np.clip(Hin, 2.0, 14.0)

# --------------------------------------------
# Modelos — Ácido con identidad explícita UGM
# --------------------------------------------
# Calcula requerimiento de ácido por UGM (usando sus extremos) y luego mezcla por proporción
acid_A = (4.0 + 1.8*extA["CaCO3"] + 1.1*extA["CuS"] + 0.00008*extA["P80"] + 0.06*extA["Hum_in"] + 0.24*extA["CaCO3"]*extA["CuS"])  # mayor base
acid_B = (1.6 + 0.9*extB["CaCO3"] + 0.5*extB["CuS"] + 0.00004*extB["P80"] + 0.03*extB["Hum_in"] + 0.12*extB["CaCO3"]*extB["CuS"])  # menor base

# Suaviza con pequeña dependencia a propiedades mezcladas para que responda con P80/Hin
acid_mix_term = 0.00002*P80 + 0.03*CaC + 0.02*CuS + 0.02*Hin
Acid = (share_A*acid_A + share_B*acid_B) + acid_mix_term
Acid = np.clip(Acid, 0, 35)

# Refino (kg/t) — mantiene lógica basada en propiedades mezcladas
Ref_lpt = (0.8 + 4.0*CuT + 2.2*CuS + 35000.0/np.maximum(P80, 2000) + 0.25*np.maximum(hum_target - Hin, 0))
Ref_kgpt = Ref_lpt * rho_refino

# Balance agua-humedad
ms = (100 - Hin)/100.0
mw_in = Hin/100.0
mw_ref = Ref_kgpt/1000.0
extra_w = (hum_target/100.0)*(ms + mw_in) - (mw_in + mw_ref)
extra_w = np.maximum(extra_w, 0.0)
Wat_kgpt = extra_w * 100.0 * 10.0  # = extra_w*1000 (kg/t)
Hum_out = 100.0 * (mw_in + mw_ref + extra_w) / (ms + mw_in + mw_ref + extra_w)

T = pd.DataFrame({
    "timestamp": idx,
    "CuT": CuT,
    "CuS": CuS,
    "CaCO3": CaC,
    "P80": P80,
    "Hum_in": Hin,
    "Acid_kgpt": Acid,
    "Refino_kgpt": Ref_kgpt,
    "Agua_kgpt": Wat_kgpt,
    "Hum_out_pct": Hum_out,
    "%UGM_A": share_A_pct,
    "%UGM_B": share_B_pct,
})

# --------------------------------------------------------------
# Estado de animación en sesión
# --------------------------------------------------------------
if k("i") not in st.session_state:
    st.session_state[k("i")] = 0
if k("playing") not in st.session_state:
    st.session_state[k("playing")] = False

# Controles de reproducción
btn_l, btn_m, btn_r = st.columns([0.15, 0.15, 0.70])
with btn_l:
    if st.button("▶ Play", key=k("play")):
        st.session_state[k("playing")] = True
with btn_m:
    if st.button("⏸ Pause", key=k("pause")):
        st.session_state[k("playing")] = False

# Slider manual de frame (sincronizado)
st.session_state[k("i")] = st.slider("Frame", 0, len(T)-1, st.session_state[k("i")], 1, key=k("frame"))
cur = st.session_state[k("i")]

# --------------------------------------------------------------
# Figura (2 filas):
#  Fila 1 → setpoints (kg/t) + Hum_out (%)
#  Fila 2 → composición UGM_A/UGM_B (área apilada 0..100%)
# --------------------------------------------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                    vertical_spacing=0.12)

# Fila 1
fig.add_trace(go.Scatter(name="Ácido (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Acid_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["magenta"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Refino (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Refino_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["blue"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Agua (kg/t)", x=T["timestamp"].iloc[:cur+1], y=T["Agua_kgpt"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["green"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Humedad_out (%)", x=T["timestamp"].iloc[:cur+1], y=T["Hum_out_pct"].iloc[:cur+1],
                         mode="lines", line=dict(color=C["silver"], width=2, dash="dot")), row=1, col=1, secondary_y=True)

# Fila 2 — composición (área apilada)
fig.add_trace(go.Scatter(name="% UGM_A", x=T["timestamp"].iloc[:cur+1], y=T["%UGM_A"].iloc[:cur+1], mode="lines",
                         line=dict(color="#9B59B6", width=0), fill="tozeroy", stackgroup="mix"), row=2, col=1)
fig.add_trace(go.Scatter(name="% UGM_B", x=T["timestamp"].iloc[:cur+1], y=T["%UGM_B"].iloc[:cur+1], mode="lines",
                         line=dict(color="#E91E63", width=0), fill="tonexty", stackgroup="mix"), row=2, col=1)

fig.update_layout(
    height=700,
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    title_text="T1 — Setpoints (kg/t), Humedad (%) y Composición UGM",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
)
fig.update_yaxes(title_text="kg/t", secondary_y=False, row=1, col=1)
fig.update_yaxes(title_text="%", secondary_y=True, row=1, col=1)
fig.update_yaxes(title_text="Composición UGM (%)", row=2, col=1, range=[0,100])

st.plotly_chart(fig, use_container_width=True, key=k("plot"))

# Panel de KPIs
st.markdown("### Estado instantáneo (frame actual)")
row = T.iloc[cur]
K1, K2, K3, K4 = st.columns(4)
with K1:
    st.metric("% UGM_A", f"{row['%UGM_A']:.1f}%")
    st.metric("% UGM_B", f"{row['%UGM_B']:.1f}%")
with K2:
    st.metric("P80 (µm)", f"{row['P80']:.0f}")
    st.metric("Hum_in (%)", f"{row['Hum_in']:.2f}")
with K3:
    st.metric("Ácido (kg/t)", f"{row['Acid_kgpt']:.2f}")
    st.metric("Refino (kg/t)", f"{row['Refino_kgpt']:.2f}")
with K4:
    st.metric("Agua (kg/t)", f"{row['Agua_kgpt']:.2f}")
    st.metric("Humedad_out (%)", f"{row['Hum_out_pct']:.2f}")

# Bucle de reproducción (incrementa frame y rerun)
if st.session_state[k("playing")]:
    nxt = (cur + 1) % len(T)
    st.session_state[k("i")] = nxt
    time.sleep(speed_ms / 1000.0)
    st.experimental_rerun()

# Pie
st.markdown(
    f"<div style='color:{C['silver']}; font-size:12px; padding-top:6px;'>"
    f"OptiBlend® — Mini animación (96 pasos, 2 días, step 30 min). UGM_A demanda más ácido que UGM_B."
    f"</div>",
    unsafe_allow_html=True,
)
