import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ================= CONFIG =================
st.set_page_config("Airbus — Dashboard", layout="wide")

BLUE="#0B5FFF"
GREEN="#22C55E"
ORANGE="#F59E0B"
RED="#EF4444"
BG="#070B14"
CARD="#0E1627"
LINE="rgba(255,255,255,.08)"
WHITE="#FFFFFF"

# ================= STYLE =================
st.markdown(f"""
<style>
html, body, [class*="css"] {{ color: {WHITE} !important; }}

.stApp {{
  background:
    radial-gradient(900px 450px at 20% 10%, rgba(11,95,255,.18), transparent 60%),
    radial-gradient(800px 420px at 85% 12%, rgba(34,197,94,.10), transparent 55%),
    linear-gradient(180deg,#0b1220,{BG});
}}

section[data-testid="stSidebar"] {{
  background: rgba(14,22,39,.92);
  border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* ---- HEADER (plus visible) ---- */
.badge {{
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.18);
  background: rgba(255,255,255,.10);
  color:{WHITE};
  font-size:.90rem;
  font-weight:700;
  letter-spacing:.2px;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
}}

/* ---- KPI CARDS (plus contrastées) ---- */
.kpis {{
  display:grid;
  grid-template-columns: repeat(6,1fr);
  gap:10px;
  margin:10px 0 6px;
}}
.kpi {{
  background: rgba(255,255,255,.08);          /* <-- plus clair */
  border:1px solid rgba(255,255,255,.16);     /* <-- bordure plus visible */
  border-radius:16px;
  padding:12px 14px;
  box-shadow: 0 16px 34px rgba(0,0,0,.35);
}}
.kpi .l {{
  font-size:.86rem;
  color: rgba(255,255,255,.85);
}}
.kpi .v {{
  font-size:1.35rem;
  font-weight:900;
  color:{WHITE};
}}

/* séparation */
hr {{
  border:none;
  border-top:1px solid rgba(255,255,255,.14);
  margin:16px 0;
}}

/* Plotly texte */
svg text {{ fill: {WHITE} !important; }}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    base = Path(__file__).parent
    for p in [base/"AIRBUS_2026-01-16.txt", base/"AIRBUS_2026-01-16"]:
        if p.exists():
            df = pd.read_csv(p, sep="\t", engine="python")
            df.columns = [c.strip() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            for col in ["ouv","haut","bas","clot","vol"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            st.sidebar.success(f"📄 Données : {p.name}")
            return df.dropna(subset=["date","clot"]).sort_values("date")
    raise FileNotFoundError("AIRBUS_2026-01-16.txt introuvable")

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Filtres")
start = st.sidebar.date_input("Début", df["date"].min().date())
end   = st.sidebar.date_input("Fin", df["date"].max().date())
maF   = st.sidebar.slider("MA courte", 5, 40, 20)
maS   = st.sidebar.slider("MA longue", 30, 120, 50)
vol_w = st.sidebar.slider("Volatilité (jours)", 10, 60, 20)

data = df[(df["date"].dt.date>=start) & (df["date"].dt.date<=end)].copy()
if len(data) < 15:
    st.error("Pas assez de données")
    st.stop()

# ================= FEATURES =================
data["ret"] = data["clot"].pct_change()
data["maF"] = data["clot"].rolling(maF).mean()
data["maS"] = data["clot"].rolling(maS).mean()
data["peak"] = data["clot"].cummax()
data["dd"] = data["clot"]/data["peak"] - 1
data["volR"] = data["ret"].rolling(vol_w).std() * np.sqrt(252)

last_close = data["clot"].iloc[-1]
perf = data["clot"].iloc[-1]/data["clot"].iloc[0]-1
max_high = data["haut"].max()
avg_vol = data["vol"].mean()
vol_ann = data["ret"].std()*np.sqrt(252)
max_dd = data["dd"].min()

# ================= HEADER =================
st.markdown(f"""
<div class="badge">Airbus (CAC 40) — 16/01/2025 → 16/01/2026</div>
""", unsafe_allow_html=True)

# ================= KPIs =================
st.markdown(f"""
<div class="kpis">
  <div class="kpi"><div class="l">Dernier close</div><div class="v">{last_close:.2f} €</div></div>
  <div class="kpi"><div class="l">Performance</div><div class="v">{perf*100:.2f} %</div></div>
  <div class="kpi"><div class="l">Plus haut</div><div class="v">{max_high:.2f} €</div></div>
  <div class="kpi"><div class="l">Volume moyen</div><div class="v">{avg_vol:,.0f}</div></div>
  <div class="kpi"><div class="l">Volatilité</div><div class="v">{vol_ann*100:.2f} %</div></div>
  <div class="kpi"><div class="l">Max drawdown</div><div class="v">{max_dd*100:.2f} %</div></div>
</div>
<hr/>
""", unsafe_allow_html=True)

# ================= GRAPHES =================
def dark(fig, title, h):
    fig.update_layout(
        title=title,
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.02)",
        font=dict(color=WHITE),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40,r=20,t=60,b=40)
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)")
    return fig

# ---- Prix + MA + Régimes
data["reg"] = (data["maF"] > data["maS"]).astype(int)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["date"], y=data["clot"], name="Clôture", line=dict(color=BLUE, width=2)))
fig.add_trace(go.Scatter(x=data["date"], y=data["maF"], name=f"MA courte ({maF})", line=dict(color=GREEN)))
fig.add_trace(go.Scatter(x=data["date"], y=data["maS"], name=f"MA longue ({maS})", line=dict(color=ORANGE)))
fig.add_hline(y=max_high, line_dash="dot", line_color="rgba(255,255,255,.4)",
              annotation_text="Plus haut période", annotation_position="top left")

s = 0
for i in range(1,len(data)):
    if data["reg"].iloc[i] != data["reg"].iloc[i-1]:
        fig.add_vrect(
            x0=data["date"].iloc[s], x1=data["date"].iloc[i-1],
            fillcolor="rgba(34,197,94,.10)" if data["reg"].iloc[i-1]==1 else "rgba(239,68,68,.10)",
            line_width=0
        )
        s = i
fig.add_vrect(x0=data["date"].iloc[s], x1=data["date"].iloc[-1],
              fillcolor="rgba(34,197,94,.10)" if data["reg"].iloc[-1]==1 else "rgba(239,68,68,.10)",
              line_width=0)

st.plotly_chart(dark(fig,"Clôture + MA + régimes",420), use_container_width=True)

# ---- Volume / Drawdown / Volatilité
c1, c2 = st.columns(2)

with c1:
    v = px.bar(data, x="date", y="vol", title="Volume journalier")
    v.update_traces(marker_color=BLUE)
    st.plotly_chart(dark(v,"Volume",360), use_container_width=True)

with c2:
    d = px.area(data, x="date", y="dd", title="Drawdown")
    d.update_traces(line_color=RED, fillcolor="rgba(239,68,68,.18)")
    d.update_yaxes(tickformat=".2%")
    st.plotly_chart(dark(d,"Drawdown",360), use_container_width=True)

vol = px.line(data, x="date", y="volR", title=f"Volatilité glissante ({vol_w} jours)")
vol.update_traces(line_color=GREEN)
vol.update_yaxes(tickformat=".2%")
st.plotly_chart(dark(vol,"Volatilité",320), use_container_width=True)
