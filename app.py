# OptiBlend — Mini App Animada (ligera)
# Objetivo: mostrar solo la animación de la respuesta del modelo ante cambios de alimentación
# - 2 días, paso 30 min (96 frames)
# - Un tambor (T1) para simplificar
# - Recalcula: P80, dosificación (Ácido/Refino/Agua en kg/t) y Hum_out (%)
# - Animación con Plotly frames (Play/Pause)
# --------------------------------------------------------------

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
  <p style='color:{C['silver']}; margin:0;'>Respuesta del modelo: P80, dosificación (kg/t) y Humedad_out (%)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Controles mínimos
# -----------------------------
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    hum_target = st.slider("Humedad objetivo (%)", 6.0, 12.0, 8.5, 0.1, key="hum_target_min")
with col2:
    rho_refino = st.slider("ρ refino (kg/L)", 0.98, 1.10, 1.02, 0.01, key="rho_refino_min")
with col3:
    mix_amp = st.slider("Amplitud cambio mezcla", 0.0, 1.0, 0.4, 0.05, key="mix_amp_min")
with col4:
    seed = st.number_input("Seed", 0, 9999, 23, 1, key="seed_min")

# -----------------------------
# Simulación ligera (un tambor)
# -----------------------------
np.random.seed(int(seed))
periods = 96  # 2 días * 48 pasos/día
freq_minutes = 30
start = pd.Timestamp.now().floor(f"{freq_minutes}min") - pd.Timedelta(minutes=freq_minutes*(periods-1))
idx = pd.date_range(start=start, periods=periods, freq=f"{freq_minutes}min")

# Señal de mezcla que modula las leyes (0..1)
phi = np.linspace(0, 2*np.pi, periods)
blend_signal = 0.5 + mix_amp * 0.5 * np.sin(phi)  # dentro de [0,1]

# Leyes base (dos extremos) que se mezclan
extA = {"CuT":0.40, "CuS":0.12, "CaCO3":2.0, "P80":9000, "Hum_in":6.5}
extB = {"CuT":0.90, "CuS":0.45, "CaCO3":6.0, "P80":6000, "Hum_in":5.0}

# Interpolación entre A y B según blend_signal
def interp(a, b, w):
    return a*(1-w) + b*w

CuT = interp(extA["CuT"],  extB["CuT"],  blend_signal) + np.random.normal(0, 0.03, periods)
CuS = interp(extA["CuS"],  extB["CuS"],  blend_signal) + np.random.normal(0, 0.02, periods)
CaC = interp(extA["CaCO3"],extB["CaCO3"],blend_signal) + np.random.normal(0, 0.4, periods)
P80 = interp(extA["P80"],  extB["P80"],  blend_signal) + np.random.normal(0, 300, periods)
Hin = interp(extA["Hum_in"],extB["Hum_in"],blend_signal) + np.random.normal(0, 0.3, periods)

# Limites razonables
CuT = np.clip(CuT, 0.05, 2.0)
CuS = np.clip(CuS, 0.0, 1.0)
CaC = np.clip(CaC, 0.0, 10.0)
P80 = np.clip(P80, 2000, 20000)
Hin = np.clip(Hin, 2.0, 14.0)

# Modelos de dosificación (kg/t) y humedad
Acid = (2.0 + 1.4*CaC + 0.8*CuS + 0.00006*P80 + 0.05*Hin + 0.18*CaC*CuS)
Acid = np.clip(Acid, 0, 30)

Ref_lpt = (0.8 + 4.0*CuT + 2.5*CuS + 35000.0/np.maximum(P80, 2000) + 0.2*np.maximum(hum_target - Hin, 0))
Ref_kgpt = Ref_lpt * rho_refino

ms = (100 - Hin)/100.0
mw_in = Hin/100.0
mw_ref = Ref_kgpt/1000.0

# Agua necesaria para objetivo (simplificado, sin evaporación)
extra_w = (hum_target/100.0)*(ms + mw_in) - (mw_in + mw_ref)
extra_w = np.maximum(extra_w, 0.0)
Wat_kgpt = extra_w * 1000.0

# Humedad de salida (sin evaporación)
Hum_out = 100.0 * (mw_in + mw_ref + extra_w) / (ms + mw_in + mw_ref + extra_w)

# DataFrame tidy
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
})

# -----------------------------
# Figura con frames (histórico acumulado hasta el frame)
# -----------------------------
fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

# Trazas base (vacías, se llenan con frames)
fig.add_trace(go.Scatter(name="Ácido (kg/t)", line=dict(color=C["magenta"], width=2)), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Refino (kg/t)", line=dict(color=C["blue"], width=2)),   row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Agua (kg/t)",   line=dict(color=C["green"], width=2)),  row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(name="Humedad_out (%)", line=dict(color=C["silver"], width=2, dash="dot")), row=1, col=1, secondary_y=True)

frames = []
for i in range(len(T)):
    fr = go.Frame(
        name=str(i),
        data=[
            go.Scatter(x=T["timestamp"].iloc[:i+1], y=T["Acid_kgpt"].iloc[:i+1]),
            go.Scatter(x=T["timestamp"].iloc[:i+1], y=T["Refino_kgpt"].iloc[:i+1]),
            go.Scatter(x=T["timestamp"].iloc[:i+1], y=T["Agua_kgpt"].iloc[:i+1]),
            go.Scatter(x=T["timestamp"].iloc[:i+1], y=T["Hum_out_pct"].iloc[:i+1]),
        ]
    )
    frames.append(fr)

fig.frames = frames

# Layout y botones de animación
fig.update_layout(
    height=520,
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
    title_text="T1 — Setpoints (kg/t) y Humedad (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": 1.18,
            "buttons": [
                {"label": "▶ Play", "method": "animate", "args": [[str(i) for i in range(len(T))], {"frame": {"duration": 80, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
            ],
        }
    ],
    sliders=[{
        "active": 0,
        "y": 0,
        "x": 0,
        "len": 1.0,
        "pad": {"b": 0, "t": 0},
        "currentvalue": {"prefix": "Frame: "},
        "steps": [{"label": str(i), "method": "animate", "args": [[str(i)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]} for i in range(len(T))]
    }]
)

fig.update_yaxes(title_text="kg/t", secondary_y=False)
fig.update_yaxes(title_text="%", secondary_y=True)

st.plotly_chart(fig, use_container_width=True, key="plot_anim_min")

# Pie
st.markdown(
    f"<div style='color:{C['silver']}; font-size:12px; padding-top:6px;'>"
    f"OptiBlend® — Mini animación (96 frames, 2 días, paso 30 min). Ajusta sliders y vuelve a reproducir."
    f"</div>",
    unsafe_allow_html=True,
)
