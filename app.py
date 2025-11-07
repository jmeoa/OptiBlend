# OptiBlend — Subplot 1×5 Animado (ISA-101) — FINAL
# Único archivo (app.py) listo para Streamlit Cloud
# - Animación Play/Pause + slider de frame (sin caché)
# - Logo fijo desde repo: assets/accenture.png
# - Ecuación del modelo en LaTeX (sidebar y versión con coeficientes)
# - Subplot 1×5 con títulos por gráfico, ejes profesionales y leyendas controladas

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
# Configura visual (ISA-101)
# ------------------------------
st.set_page_config(page_title="OptiBlend — 1×5 Animado (ISA-101)", layout="wide", page_icon="⚙️")
pio.templates.default = "simple_white"

st.markdown(
    """
    <style>
      .stApp, .block-container { background: #F3F4F7 !important; }
      .metric { border: 1px solid #eee; border-radius: 10px; padding: 6px; }
      .acc-header { display:flex; align-items:center; gap:16px; border-bottom:1px solid #E0E0E0; padding:10px 0 14px 0; }
      .acc-title  { font-weight:800; font-size:28px; color:#111; line-height:1.05; }
      .acc-sub    { color:#555; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Namespace único para keys (evita colisiones)
if "app_ns" not in st.session_state:
    st.session_state["app_ns"] = f"opb_{uuid4().hex[:8]}"
NS = st.session_state["app_ns"]
k = lambda name: f"{NS}:{name}"

# ------------------------------
# Header con logo fijo desde repo + título
# ------------------------------
c1, c2 = st.columns([0.14, 0.86])
with c1:
    LOGO_PATH = "assets/accenture.png"
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=160)
with c2:
    st.markdown(
        """
        <div class="acc-header">
          <div>
            <div class="acc-title">OptiBlend® — 1×5 Animated Dashboard</div>
            <div class="acc-sub">ISA-101 • Mezcla UGM → P80/−100# / TR → Leyes → Dosificación → Hum_total</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# Controles (sidebar)
# ------------------------------
st.sidebar.header("Controles")

# Polinomio Ácido (kg/t) en LaTeX (forma general)
st.sidebar.markdown("**Polinomio Ácido (kg/t):**")
st.sidebar.latex(
    r"""
    \mathrm{Acid} = a_0 + a_1\,\mathrm{CaCO_3}
    + a_2\,\mathrm{CuS} + a_3\,P80_{\mathrm{mm}}
    + a_4\,\mathrm{Hum_{nat}} + a_5\,\mathrm{Finos_{-100\#}}
    + a_6\,(\mathrm{CaCO_3}\cdot \mathrm{CuS})
    """
)

# Coeficientes (rango calibrado para 5–40 kg/t típico)
a0 = st.sidebar.slider("a0", 0.0, 10.0, 5.0, 0.1, key=k("a0"))
a1 = st.sidebar.slider("a1 (× CaCO3)", 0.0, 3.0, 1.2, 0.1, key=k("a1"))
a2 = st.sidebar.slider("a2 (× CuS)", 0.0, 5.0, 2.0, 0.1, key=k("a2"))
a3 = st.sidebar.slider("a3 (× P80 mm)", 0.0, 1.0, 0.30, 0.01, key=k("a3"))
a4 = st.sidebar.slider("a4 (× HumNat %)", 0.0, 2.0, 0.30, 0.01, key=k("a4"))
a5 = st.sidebar.slider("a5 (× Finos −100# %)", 0.0, 2.0, 0.40, 0.01, key=k("a5"))
a6 = st.sidebar.slider("a6 (× CaCO3·CuS)", 0.0, 1.0, 0.25, 0.01, key=k("a6"))

n_drum = st.sidebar.number_input("Nº Tambores", min_value=1, max_value=4, value=2, step=1, key=k("ndr"))
n_ugm  = st.sidebar.number_input("Nº UGM", min_value=2, max_value=5, value=3, step=1, key=k("nugm"))
hum_obj = st.sidebar.slider("Humedad producto objetivo (%)", 5.0, 15.0, 10.0, 0.1, key=k("humobj"))

# Animación
speed_ms = st.sidebar.slider("Velocidad (ms/frame)", 30, 400, 120, 10, key=k("speed"))
seed = st.sidebar.number_input("Seed", 0, 9999, 42, 1, key=k("seed"))

# ------------------------------
# Simulación (rangos pedidos)
# ------------------------------
np.random.seed(int(seed))
periods = 96  # 2 días @ 30 min
idx = pd.date_range(pd.Timestamp.now().floor("30min") - pd.Timedelta(minutes=30*(periods-1)),
                    periods=periods, freq="30min")

# Mezcla de UGM (proporciones que suman 1) — Dirichlet
if int(n_ugm) == 3:
    alpha = np.array([2.0, 3.0, 1.6])
else:
    alpha = np.ones(int(n_ugm)) + 0.5
mix = np.random.dirichlet(alpha, size=periods)  # (periods, n_ugm)

# Propiedades base por UGM (dentro de rangos)
# CuT 0.2–1.0 %; CuS 0.2–0.8 %; CaCO3 1–10 %; P80 10–15 mm; −100# 5–20 %; HumNat 1–3 %
ugm_props = []
for j in range(int(n_ugm)):
    ugm_props.append(dict(
        CuT   = np.random.uniform(0.25, 0.95),
        CuS   = np.random.uniform(0.25, 0.75),
        CaCO3 = np.random.uniform(1.5, 9.0),
        P80mm = np.random.uniform(10.2, 14.5),
        Finos = np.random.uniform(6.0, 19.0),
        HumNat= np.random.uniform(1.2, 2.8),
    ))

props_arr = np.array([[p["CuT"], p["CuS"], p["CaCO3"], p["P80mm"], p["Finos"], p["HumNat"]] for p in ugm_props])
blend = mix @ props_arr
CuT_blend, CuS_blend, CaCO3_blend, P80_blendmm, Finos_blend, HumN_blend = blend.T

# Soft-sensor: Tiempo de residencia (TR, min) y efecto en P80 y Finos
TR = 8 + 4*np.sin(np.linspace(0, 2*np.pi, periods))  # 8–12 min
P80_effmm = np.clip(P80_blendmm * (1.0 - 0.03*(TR-10)), 10.0, 15.0)
Finos_eff = np.clip(Finos_blend * (1.0 + 0.02*(TR-10)), 5.0, 25.0)

# Tonelaje por tambor (tph)
feeds = {}
for d in range(int(n_drum)):
    base = np.random.uniform(800, 1200)
    noise = 120*np.sin(np.linspace(0, 3*np.pi, periods) + d)
    feeds[f"T{d+1}"] = np.clip(base + noise + np.random.normal(0, 30, periods), 500, 1500)

# Dosificaciones y humedad
Acid = a0 + a1*CaCO3_blend + a2*CuS_blend + a3*P80_effmm + a4*HumN_blend + a5*Finos_eff + a6*(CaCO3_blend*CuS_blend)
Acid = np.clip(Acid, 5, 40)

Refino = np.clip(2.0 + 6.0*CuT_blend + 2.5*CuS_blend + 0.2*TR + 0.6*np.maximum(hum_obj - HumN_blend, 0), 0, 40)

agua_req_kgpt = hum_obj * 10  # ~kg/t por % objetivo (aprox)
Agua = np.clip(np.maximum(agua_req_kgpt - Refino*0.95 - HumN_blend*10, 0), 0, 40)

Hum_out = np.clip(HumN_blend + 0.095*Refino + 0.10*Agua, 5, 18)

# ------------------------------
# Estado de animación
# ------------------------------
if k("i") not in st.session_state:
    st.session_state[k("i")] = 0
if k("playing") not in st.session_state:
    st.session_state[k("playing")] = False

label = "⏸ Pause" if st.session_state[k("playing")] else "▶ Play"
if st.button(label, key=k("pp")):
    st.session_state[k("playing")] = not st.session_state[k("playing")]

st.session_state[k("i")] = st.slider("Frame", 0, periods-1, st.session_state[k("i")], 1, key=k("frame"))
cur = int(st.session_state[k("i")])

# ------------------------------
# Figura 1×5 con autoescala
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

# Paleta sobria (ISA-101 / acentos discretos)
COL = {
    "ugm": ["#7E57C2", "#D81B60", "#8E24AA", "#1E88E5", "#43A047"],
    "tph": ["#455A64", "#9E9E9E", "#607D8B", "#78909C"],
    "p80": "#F9A825",
    "f100": "#6D4C41",
    "tr":  "#616161",
    "acid":  "#A100FF",
    "agua":  "#2ECC71",
    "refino":"#0072CE",
    "hum":   "#2B2B2B",
}

# (1) Mezcla UGM (stackgroup) — leyenda visible
for j in range(int(n_ugm)):
    fig.add_trace(
        go.Scatter(
            x=idx[:cur+1], y=100*mix[:cur+1, j], mode="lines",
            name=f"UGM{j+1}", stackgroup="ugm",
            line=dict(width=1.6, color=COL["ugm"][j % len(COL["ugm"])]),
            showlegend=True
        ),
        row=1, col=1
    )
fig.update_yaxes(title_text="Mezcla UGM (%)", row=1, col=1, autorange=True)

# (2) TPH por tambor — sin leyenda (evita ruido)
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

# (3) P80 + −100# + TR — P80 al eje izq; −100# y TR al derecho
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=P80_effmm[:cur+1], name="P80 (mm)", mode="lines",
               line=dict(width=2, color=COL["p80"]), showlegend=True),
    row=1, col=3, secondary_y=False
)
fig.add_trace(
    go.Scatter(x=idx[:cur+1], y=Finos_eff[:cur+1], name="−100# (%)", mode="lines",
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

# (4) Leyes — todas al eje derecho (sin leyenda para no saturar)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=100*CuT_blend[:cur+1], name="CuT %",
                         mode="lines", line=dict(width=1.6, color="#424242"),
                         showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=100*CuS_blend[:cur+1], name="CuS %",
                         mode="lines", line=dict(width=1.2, dash="dot", color="#7B1FA2"),
                         showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=CaCO3_blend[:cur+1], name="CaCO3 %",
                         mode="lines", line=dict(width=1.2, dash="dash", color="#1565C0"),
                         showlegend=False), row=1, col=4, secondary_y=True)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=HumN_blend[:cur+1], name="HumNat %",
                         mode="lines", line=dict(width=1.2, color="#00897B"),
                         showlegend=False), row=1, col=4, secondary_y=True)
fig.update_yaxes(title_text="%", row=1, col=4, secondary_y=True, autorange=True)
fig.update_yaxes(title_text="Leyes", row=1, col=4, secondary_y=False, showgrid=False)

# (5) Dosificación (kg/t) + Hum_total (%) — leyenda visible aquí
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Acid[:cur+1], name="Ácido (kg/t)",
                         mode="lines", line=dict(width=2, color=COL["acid"])),
              row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Agua[:cur+1], name="Agua (kg/t)",
                         mode="lines", line=dict(width=1.8, color=COL["agua"])),
              row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Refino[:cur+1], name="Refino (kg/t)",
                         mode="lines", line=dict(width=1.8, color=COL["refino"])),
              row=1, col=5, secondary_y=False)
fig.add_trace(go.Scatter(x=idx[:cur+1], y=Hum_out[:cur+1], name="Hum_total %",
                         mode="lines", line=dict(width=1.6, dash="dot", color=COL["hum"])),
              row=1, col=5, secondary_y=True)
fig.update_yaxes(title_text="kg/t", row=1, col=5, secondary_y=False, autorange=True)
fig.update_yaxes(title_text="%", row=1, col=5, secondary_y=True, autorange=True)

# Layout general (fondo gris + leyenda visible + márgenes auto)
fig.update_layout(
    height=540,
    hovermode="x unified",
    paper_bgcolor="#F5F6FA",
    plot_bgcolor="#F7F7F9",
    font=dict(color="#111", size=12),
    legend=dict(
        orientation="h",
        y=1.10,
        x=1.0,
        xanchor="right",
        font=dict(size=11),
        tracegroupgap=8
    ),
    margin=dict(l=30, r=10, t=90, b=30)
)

# Rejillas + ejes profesionales (ISA-101)
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

st.plotly_chart(fig, use_container_width=True, key=k("plot"))

# ------------------------------
# KPIs (frame actual) — dinámicos
# ------------------------------
row = cur
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Ácido (kg/t)", f"{Acid[row]:.1f}")
    st.metric("Hum_total (%)", f"{Hum_out[row]:.1f}")
with c2:
    st.metric("Refino (kg/t)", f"{Refino[row]:.1f}")
    st.metric("Agua (kg/t)", f"{Agua[row]:.1f}")
with c3:
    st.metric("P80 (mm)", f"{P80_effmm[row]:.2f}")
    st.metric("−100# (%)", f"{Finos_eff[row]:.1f}")
with c4:
    st.metric("CuT (%)", f"{100*CuT_blend[row]:.2f}")
    st.metric("CuS (%)", f"{100*CuS_blend[row]:.2f}")
with c5:
    st.metric("CaCO3 (%)", f"{CaCO3_blend[row]:.1f}")
    st.metric("HumNat (%)", f"{HumN_blend[row]:.2f}")

# Ecuación con coeficientes actuales (LaTeX)
st.latex(
    rf"""
    \mathrm{{Acid}}\,(\mathrm{{kg/t}}) =
    {a0:.2f} + {a1:.2f}\,\mathrm{{CaCO_3}}
    + {a2:.2f}\,\mathrm{{CuS}} + {a3:.3f}\,P80_{{\mathrm{{mm}}}}
    + {a4:.3f}\,\mathrm{{Hum_{{nat}}}} + {a5:.3f}\,\mathrm{{Finos_{{-100\#}}}}
    + {a6:.3f}\,(\mathrm{{CaCO_3}}\cdot \mathrm{{CuS}})
    """
)

# ------------------------------
# Bucle de animación (Play/Pause)
# ------------------------------
if st.session_state[k("playing")]:
    st.session_state[k("i")] = (cur + 1) % periods
    time.sleep(st.session_state[k("speed")] / 1000.0)
    st.rerun()
