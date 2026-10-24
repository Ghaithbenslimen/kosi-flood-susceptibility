import os
import base64
import numpy as np
import pandas as pd
import rasterio
import joblib
import folium
import requests
from PIL import Image
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject, Resampling
from rasterio.warp import transform as warp_transform, transform_bounds
from rasterio.transform import rowcol
from streamlit_folium import st_folium
import streamlit as st

st.set_page_config(
    page_title="Kosi Atlas — Susceptibilité aux crues",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "manual_result" not in st.session_state:
    st.session_state.manual_result = None
    st.session_state.manual_snapshot = None
    st.session_state.manual_feats = None

FRIENDLY_NAMES = {
    "DEM": "Altitude", "Slope": "Pente", "Aspect": "Orientation",
    "Curvature": "Courbure du terrain", "TWI": "Humidité du sol",
    "NDVI": "Végétation", "NDBI": "Zones bâties",
    "DD": "Densité de rivières", "DR": "Réseau de drainage",
    "GFI": "Risque géologique",
}

DESCRIPTIONS = {
    "DEM": "Hauteur du terrain (m).",
    "Slope": "Inclinaison du terrain (°).",
    "Aspect": "Direction du versant (°).",
    "Curvature": "Forme du terrain.",
    "TWI": "Indice d'humidité topographique.",
    "NDVI": "Densité de végétation.",
    "NDBI": "Proportion de surfaces construites.",
    "DD": "Longueur de rivières par km².",
    "DR": "Ratio de drainage.",
    "GFI": "Indice géologique de susceptibilité.",
}

UNITS = {
    "DEM": "m", "Slope": "°", "Aspect": "°", "Curvature": "",
    "TWI": "", "NDVI": "", "NDBI": "", "DD": "km/km²", "DR": "",
    "GFI": "",
}

SLIDER_RANGES = {
    "DEM": (0.0, 500.0, 100.0, 1.0), "Slope": (0.0, 30.0, 2.0, 0.1),
    "Aspect": (0.0, 360.0, 180.0, 1.0), "Curvature": (-5.0, 5.0, 0.0, 0.1),
    "TWI": (-5.0, 25.0, 8.0, 0.1), "NDVI": (-1.0, 1.0, 0.3, 0.01),
    "NDBI": (-1.0, 1.0, 0.0, 0.01), "DD": (0.0, 10.0, 2.0, 0.1),
    "DR": (0.0, 10.0, 1.0, 0.1), "GFI": (0.0, 10.0, 1.0, 0.1),
}

RAW_FEATURES = ["DEM", "Slope", "Aspect", "Curvature", "TWI",
                "NDVI", "NDBI", "DD", "DR", "GFI"]

CLASS_LABELS = {0: "Hors zone", 1: "Très faible", 2: "Faible", 3: "Élevé", 4: "Très élevé"}

MODEL_METRICS = {
    "test": {"AUC-ROC": 0.927, "Precision": 0.9278, "Recall": 0.9208,
             "F1": 0.9243, "Accuracy": 0.9245},
}

STACKING_COMPONENTS = [
    ("01", "RandomForest", "200 arbres", "class_weight balanced"),
    ("02", "Support Vector", "Noyau RBF", "probabilités calibrées"),
    ("03", "XGBoost", "200 estimateurs", "gradient boosting"),
    ("04", "Logistic Regression", "Méta-modèle", "agrégation des probas"),
]


@st.cache_data(show_spinner=False)
def load_hero_image():
    CANDIDATES = [
        "assets/kosi1.jpg", "assets/kosi2.jpg", "assets/megafan.jpeg",
        "assets/megafan.png", "assets/megafan.jpg",
        "data/images/megafan.png", "data/images/megafan.jpg",
        "outputs/megafan.png", "outputs/megafan.jpg",
        "data/raw/megafan.png", "data/raw/megafan.jpg",
    ]
    for p in CANDIDATES:
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = "png" if p.lower().endswith(".png") else "jpeg"
            return f"data:image/{ext};base64,{b64}"
    return None

hero_uri = load_hero_image()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --paper: #FAF7F2; --paper-raised: #FDFBF8;
  --ink: #1A2B24; --ink-soft: #2D3D35; --muted: #8A8578;
  --line: #E3DCD0; --line-strong: #C9C0B0;
  --forest: #1F3D33; --terracotta: #B85C38; --gold: #C9A961; --sage: #7BA690;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: var(--ink);
  font-feature-settings: "tnum" 1, "lnum" 1;
}

[data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stSidebar"] { background: var(--forest); border-right: 1px solid rgba(255,255,255,0.05); }
[data-testid="stSidebar"] * { color: #E8DFD1 !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
  color: var(--gold) !important;
  font-family: 'Fraunces', serif !important;
  font-weight: 400 !important;
}

.block-container { max-width: 1400px; padding: 2rem 3rem 4rem; }

h1, h2, h3 {
  font-family: 'Fraunces', serif !important;
  color: var(--ink) !important;
  font-weight: 400 !important;
  letter-spacing: -0.025em !important;
}
h1 { font-size: 3.2rem !important; line-height: 1.02 !important; }
h2 { font-size: 1.5rem !important; font-weight: 500 !important; }
h3 { font-size: 1.15rem !important; font-weight: 500 !important; }

/* --- BOUTONS --- */
.stButton > button, .stDownloadButton > button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  border-radius: 2px !important;
  padding: 1rem 2rem !important;
  transition: all 0.25s ease !important;
  min-height: 3rem !important;
}
.stButton > button[kind="secondary"], .stDownloadButton > button {
  background: var(--paper-raised) !important;
  color: var(--ink) !important;
  border: 1.5px solid var(--ink) !important;
}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
  background: var(--ink) !important;
  color: var(--paper) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--forest) 0%, var(--terracotta) 100%) !important;
  color: var(--paper) !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(31,61,51,0.25), inset 0 0 0 1px rgba(255,255,255,0.1) !important;
  font-size: 0.9rem !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(184,92,56,0.35), inset 0 0 0 1px rgba(255,255,255,0.15) !important;
}

/* --- TABS --- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  border-bottom: 2px solid var(--line) !important;
  gap: 0 !important;
  background: var(--paper-raised) !important;
  padding: 0 0.5rem !important;
  margin-bottom: 2rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'Fraunces', serif !important;
  font-size: 1.05rem !important;
  font-weight: 500 !important;
  padding: 1.2rem 2.2rem !important;
  color: var(--muted) !important;
  border-radius: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--ink) !important;
  background: rgba(31,61,51,0.03) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 3px solid var(--terracotta) !important;
  background: var(--paper) !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

/* --- MASTHEAD --- */
.masthead {
  position: relative;
  border-bottom: 2px solid var(--ink);
  padding: 2.5rem 2.5rem 2rem;
  margin-bottom: 2.5rem;
  overflow: hidden;
  min-height: 240px;
  display: flex;
  align-items: end;
  background: var(--paper);
}
.masthead::before {
  content: "";
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  opacity: 0.30;
  z-index: 0;
}
.masthead-inner {
  position: relative;
  z-index: 1;
  width: 100%;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: end;
  gap: 3rem;
}
.masthead-kicker {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--terracotta);
  margin-bottom: 0.9rem;
  display: block;
}
.masthead h1 { margin: 0; }
.masthead-meta {
  text-align: right;
  font-size: 0.78rem;
  color: var(--ink-soft);
  line-height: 1.7;
  border-left: 1px solid var(--line-strong);
  padding-left: 1.5rem;
}
.masthead-meta strong {
  color: var(--ink);
  font-weight: 600;
  display: block;
  font-size: 0.68rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}

/* --- SECTIONS --- */
.section-kicker {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--line);
  padding-bottom: 0.7rem;
}
.section-num {
  font-family: 'Fraunces', serif;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--terracotta);
}
.section-title {
  font-family: 'Fraunces', serif;
  font-size: 1.4rem;
  font-weight: 500;
  color: var(--ink);
  margin: 0;
}
.section-note {
  margin-left: auto;
  font-size: 0.74rem;
  color: var(--muted);
  font-style: italic;
}

/* --- PERFORMANCE --- */
.perf-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  background: var(--paper-raised);
  border: 1px solid var(--line);
}
.perf-cell {
  padding: 1.6rem 1.4rem;
  border-right: 1px solid var(--line);
  position: relative;
}
.perf-cell:last-child { border-right: none; }
.perf-cell::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--sage), var(--gold), var(--terracotta));
  opacity: 0.8;
}
.perf-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.8rem;
}
.perf-value {
  font-family: 'Fraunces', serif;
  font-size: 2.4rem;
  font-weight: 400;
  color: var(--ink);
  line-height: 1;
}

/* --- PIPELINE --- */
.pipeline {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--line);
  background: var(--paper-raised);
}
.pipe-step {
  padding: 1.8rem 1.4rem;
  border-right: 1px solid var(--line);
  position: relative;
}
.pipe-step:last-child { border-right: none; }
.pipe-step:not(:last-child)::after {
  content: "→";
  position: absolute;
  right: -0.65rem;
  top: 50%;
  transform: translateY(-50%);
  background: var(--paper-raised);
  color: var(--terracotta);
  font-family: 'Fraunces', serif;
  font-size: 1.2rem;
  padding: 0.2rem 0.4rem;
  z-index: 2;
}
.pipe-num {
  font-family: 'Fraunces', serif;
  font-size: 0.8rem;
  color: var(--terracotta);
  font-weight: 500;
  margin-bottom: 0.8rem;
  display: block;
}
.pipe-name {
  font-family: 'Fraunces', serif;
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--ink);
  display: block;
  margin-bottom: 0.4rem;
}
.pipe-detail {
  font-size: 0.74rem;
  color: var(--muted);
  line-height: 1.5;
  display: block;
}

/* --- DISTRIBUTION RISQUE --- */
.risk-distribution {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  background: var(--paper-raised);
  border: 1px solid var(--line);
}
.risk-dist-cell {
  padding: 1.8rem 1.4rem;
  border-right: 1px solid var(--line);
  position: relative;
}
.risk-dist-cell:last-child { border-right: none; }
.risk-dist-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.9rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.risk-dist-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.risk-dist-count {
  font-family: 'Fraunces', serif;
  font-size: 2rem;
  font-weight: 400;
  color: var(--ink);
  line-height: 1;
  letter-spacing: -0.02em;
}
.risk-dist-unit {
  font-size: 0.74rem;
  color: var(--muted);
  margin-top: 0.5rem;
  font-style: italic;
}
.risk-dist-pct {
  font-size: 0.9rem;
  color: var(--terracotta);
  font-weight: 600;
  margin-top: 0.6rem;
  letter-spacing: 0.02em;
}
.risk-dist-total {
  margin-top: 1rem;
  padding: 1.1rem 1.4rem;
  background: var(--paper-raised);
  border: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.risk-dist-total-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
}
.risk-dist-total-value {
  font-family: 'Fraunces', serif;
  font-size: 1.3rem;
  color: var(--ink);
}

/* --- PANELS --- */
.panel {
  background: var(--paper-raised);
  border: 1px solid var(--line);
  padding: 1.5rem 1.7rem;
  margin-bottom: 1.2rem;
}
.panel-kicker {
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.panel-title {
  font-family: 'Fraunces', serif;
  font-size: 1.25rem;
  font-weight: 500;
  color: var(--ink);
  margin: 0;
}

/* --- RISK CARD --- */
.risk-card {
  background: var(--forest);
  color: #E8DFD1;
  padding: 1.8rem;
  border-radius: 2px;
}
.risk-card-kicker {
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.9rem;
  display: block;
}
.risk-card-loc {
  font-family: 'Fraunces', serif;
  font-size: 1.35rem;
  font-weight: 400;
  color: #F5EFE5;
  margin: 0 0 0.4rem 0;
  line-height: 1.2;
}
.risk-card-coords {
  font-size: 0.76rem;
  color: #A8B5AE;
  margin-bottom: 1.5rem;
  letter-spacing: 0.05em;
}
.risk-label-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.6rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(255,255,255,0.1);
}
.risk-level-name {
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-size: 1rem;
  color: var(--gold);
}
.risk-num {
  font-family: 'Fraunces', serif;
  font-size: 3.5rem;
  font-weight: 300;
  color: #F5EFE5;
  line-height: 0.9;
  letter-spacing: -0.03em;
}
.spectrum {
  position: relative;
  margin: 1.6rem 0 0.8rem 0;
}
.spectrum-track {
  height: 6px;
  background: linear-gradient(90deg, #7BA690 0%, #7BA690 25%, #C9A961 25%, #C9A961 55%, #C9704A 55%, #C9704A 80%, #8B2E2E 80%, #8B2E2E 100%);
  border-radius: 1px;
}
.spectrum-marker {
  position: absolute;
  top: -6px;
  width: 3px;
  height: 18px;
  background: #F5EFE5;
  transform: translateX(-50%);
  box-shadow: 0 0 0 2px var(--forest);
}
.spectrum-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.68rem;
  color: #A8B5AE;
  margin-top: 0.6rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* --- CONSISTENCY --- */
.consistency-ok {
  background: rgba(123,166,144,0.1);
  border-left: 3px solid var(--sage);
  padding: 0.7rem 1rem;
  font-size: 0.78rem;
  color: var(--ink-soft);
  margin: 1rem 0;
}
.consistency-warn {
  background: rgba(201,169,97,0.12);
  border-left: 3px solid var(--gold);
  padding: 0.7rem 1rem;
  font-size: 0.78rem;
  color: var(--ink-soft);
  margin: 1rem 0;
}
.consistency-stale {
  background: rgba(184,92,56,0.08);
  border-left: 3px solid var(--terracotta);
  padding: 0.7rem 1rem;
  font-size: 0.78rem;
  color: var(--ink-soft);
  margin: 1rem 0;
}

/* --- FEATURES --- */
.feature-line {
  padding: 0.85rem 0;
  border-bottom: 1px solid var(--line);
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: baseline;
  gap: 1rem;
}
.feature-line:last-child { border-bottom: none; }
.feature-name { font-size: 0.85rem; color: var(--ink-soft); font-weight: 500; }
.feature-val { font-family: 'Fraunces', serif; font-size: 1rem; color: var(--ink); font-weight: 500; }
.feature-bar {
  grid-column: 1 / -1;
  height: 3px;
  background: var(--line);
  margin-top: 0.5rem;
  overflow: hidden;
}
.feature-bar span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, var(--sage), var(--forest));
}

/* --- MINI KPI --- */
.mini-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--line);
  background: var(--paper-raised);
  margin-bottom: 1.5rem;
}
.mini-kpi-cell {
  padding: 1.2rem 1.3rem;
  border-right: 1px solid var(--line);
  position: relative;
}
.mini-kpi-cell:last-child { border-right: none; }
.mini-kpi-cell::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
}
.mini-kpi-lbl {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
  display: block;
}
.mini-kpi-num {
  font-family: 'Fraunces', serif;
  font-size: 1.7rem;
  font-weight: 400;
  color: var(--ink);
  display: block;
  line-height: 1;
}
.mini-kpi-pct {
  font-size: 0.74rem;
  color: var(--muted);
  margin-top: 0.3rem;
  display: block;
}

.sidebar-legend-item {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.5rem 0;
  font-size: 0.8rem;
}
.legend-bar {
  width: 32px;
  height: 3px;
  border-radius: 1px;
}

[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; }
.stSlider label { font-size: 0.82rem !important; color: var(--ink-soft) !important; }

[data-testid="stFileUploader"] {
  border: 1.5px dashed var(--line-strong) !important;
  border-radius: 2px !important;
  background: var(--paper-raised) !important;
  padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

if hero_uri:
    st.markdown(f"""
    <style>
    .masthead::before {{ background-image: url('{hero_uri}'); }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .masthead::before {
        background-image: linear-gradient(135deg, rgba(31,61,51,0.06), rgba(184,92,56,0.06));
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# CHARGEMENT
# ============================================================
@st.cache_resource
def load_model():
    return (
        joblib.load("data/processed/stacking_model_improvedV2.pkl"),
        joblib.load("data/processed/scaler_improvedV2.pkl"),
        joblib.load("data/processed/feature_names_improvedV2.pkl"),
        joblib.load("data/processed/rank_distributions.pkl"),
    )

@st.cache_resource
def load_rasters():
    srcs = {f: rasterio.open(f"data/raw/{f}.tif") for f in RAW_FEATURES}
    with rasterio.open("data/raw/DEM.tif") as ref:
        return srcs, ref.crs, ref.transform, ref.bounds

@st.cache_resource
def load_class_map():
    CANDIDATES = [
        "outputs/flood_susceptibility_class_kosi_v6_50m.tif",
        "outputs/flood_susceptibility_class_kosi_v5.tif",
        "outputs/flood_susceptibility_class_kosi_v4.tif",
        "outputs/flood_susceptibility_class_kosi_v3.tif",
        "outputs/flood_susceptibility_class_kosi.tif",
    ]
    CLASS_TIF = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if CLASS_TIF is None:
        return None
    with rasterio.open(CLASS_TIF) as src:
        return {"arr": src.read(1), "crs": src.crs, "transform": src.transform,
                "shape": src.shape, "bounds": src.bounds, "path": CLASS_TIF}

@st.cache_resource
def compute_class_distribution(_cm):
    """Calcule le nombre de pixels et la surface par classe."""
    if _cm is None:
        return None
    arr = _cm["arr"]
    left, bottom, right, top = _cm["bounds"]
    h, w = _cm["shape"]
    width_km = (right - left) / 1000.0
    height_km = (top - bottom) / 1000.0
    pixel_km2 = (width_km * height_km) / (w * h)
    counts = {cls: int((arr == cls).sum()) for cls in [1, 2, 3, 4]}
    total = sum(counts.values())
    areas = {cls: counts[cls] * pixel_km2 for cls in [1, 2, 3, 4]}
    total_area = total * pixel_km2
    return {"counts": counts, "areas": areas, "total": total,
            "total_area": total_area, "pixel_km2": pixel_km2}

@st.cache_resource
def generate_colored_overlay():
    cm = load_class_map()
    if cm is None:
        raise FileNotFoundError("Aucune carte classifiée dans outputs/.")
    left, bottom, right, top = transform_bounds(cm["crs"], "EPSG:4326", *cm["bounds"])
    h, w = cm["arr"].shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    COLORS = {1: (123, 166, 144, 210), 2: (201, 169, 97, 210),
              3: (201, 112, 74, 210), 4: (139, 46, 46, 220)}
    for cls, color in COLORS.items():
        rgba[cm["arr"] == cls] = color
    rgba = np.flipud(rgba)
    os.makedirs("outputs", exist_ok=True)
    img_path = "outputs/overlay_colored_map.png"
    Image.fromarray(rgba, mode="RGBA").save(img_path, optimize=True)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}", (left, bottom, right, top), cm["path"]


def get_percentile(values, sorted_ref):
    return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)

def build_features(feats, ranks):
    DEM, Slope = feats["DEM"], feats["Slope"]
    TWI, DD = feats["TWI"], feats["DD"]
    NDVI, NDBI = feats["NDVI"], feats["NDBI"]
    out = dict(feats)
    out["DEM_Slope"] = DEM / (Slope + 1)
    out["TWI_DD"] = TWI * DD
    out["NDVI_NDBI"] = NDVI - NDBI
    out["Vulnerability"] = (-get_percentile(DEM, ranks["DEM"])
                            + get_percentile(TWI, ranks["TWI"])
                            + get_percentile(DD, ranks["DD"])
                            - get_percentile(NDVI, ranks["NDVI"]))
    out["Roughness"] = 0.0
    return out

def predict_single(feats, stacking, scaler, feature_names, ranks):
    fe = build_features(feats, ranks)
    X = np.array([[fe[f] for f in feature_names]])
    return float(stacking.predict_proba(scaler.transform(X))[0, 1]), fe

def predict_batch(df, stacking, scaler, feature_names, ranks):
    rows = []
    for _, row in df.iterrows():
        fe = build_features({f: float(row[f]) for f in RAW_FEATURES}, ranks)
        rows.append([fe[f] for f in feature_names])
    return stacking.predict_proba(scaler.transform(np.array(rows)))[:, 1]

def sample_aligned_at_pixel(row, col, cm, srcs):
    win = Window(col, row, 1, 1)
    wt = window_transform(win, cm["transform"])
    feats, missing = {}, False
    for name, src in srcs.items():
        if (src.crs == cm["crs"] and src.transform == cm["transform"]
                and (src.height, src.width) == cm["shape"]):
            val = float(src.read(1, window=win)[0, 0])
        else:
            dest = np.full((1, 1), np.nan, dtype=np.float32)
            reproject(rasterio.band(src, 1), dest, dst_transform=wt,
                      dst_crs=cm["crs"], resampling=Resampling.bilinear,
                      src_nodata=src.nodata, dst_nodata=np.nan)
            val = float(dest[0, 0])
        if not np.isfinite(val):
            val, missing = 0.0, True
        feats[name] = val
    return feats, missing

@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat, lon):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": lat, "lon": lon, "format": "jsonv2",
                                 "addressdetails": 1, "zoom": 14, "accept-language": "fr"},
                         headers={"User-Agent": "KosiAtlas/1.0"}, timeout=5)
        r.raise_for_status()
        addr = r.json().get("address", {})
        parts = [addr.get("city") or addr.get("town") or addr.get("village")
                 or addr.get("suburb") or addr.get("county"),
                 addr.get("state"), addr.get("country")]
        return ", ".join(p for p in parts if p)
    except requests.exceptions.RequestException:
        return None

def risk_level(prob):
    if prob < 0.40: return "Très faible"
    if prob < 0.70: return "Faible"
    if prob < 0.90: return "Élevé"
    return "Très élevé"


try:
    stacking, scaler, feature_names, ranks = load_model()
    srcs, crs, ref_transform, ref_bounds = load_rasters()
except FileNotFoundError as e:
    st.error(f"Fichier manquant : {e}")
    st.stop()

cm = load_class_map()
dist = compute_class_distribution(cm)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="font-family:Fraunces,serif;font-size:1.5rem;color:#F5EFE5;letter-spacing:-0.02em;margin-bottom:.3rem;">Kosi Atlas</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.7rem;letter-spacing:.22em;text-transform:uppercase;color:#C9A961;margin-bottom:1.5rem;">Flood Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:#A8B5AE;margin-bottom:.7rem;">Projet</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Fraunces,serif;font-size:1rem;color:#F5EFE5;line-height:1.5;">Kosi Megafan<br><span style="font-size:.78rem;color:#A8B5AE;font-style:italic;">Bihar, Inde — 100 m</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:#A8B5AE;margin-bottom:.7rem;">Modèle</div>', unsafe_allow_html=True)
    st.metric("AUC", "0.927")
    st.metric("Rappel spatial", "76.3%")
    st.markdown("---")
    st.markdown('<div style="font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:#A8B5AE;margin-bottom:.7rem;">Carte</div>', unsafe_allow_html=True)
    show_overlay = st.checkbox("Afficher la couche colorée", value=True)
    opacity_val = st.slider("Opacité", 0.1, 1.0, 0.75, 0.05)
    st.markdown("---")
    st.markdown('<div style="font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:#A8B5AE;margin-bottom:.7rem;">Légende</div>', unsafe_allow_html=True)
    for name, gradient in [
        ("Très faible", "linear-gradient(90deg,#A8C9B5,#7BA690)"),
        ("Faible", "linear-gradient(90deg,#DFCB8A,#C9A961)"),
        ("Élevé", "linear-gradient(90deg,#D88A6A,#C9704A)"),
        ("Très élevé", "linear-gradient(90deg,#A84A4A,#8B2E2E)"),
    ]:
        st.markdown(f'<div class="sidebar-legend-item"><span class="legend-bar" style="background:{gradient};"></span><span>{name}</span></div>', unsafe_allow_html=True)

# ============================================================
# MASTHEAD
# ============================================================
st.markdown("""
<div class="masthead">
  <div class="masthead-inner">
    <div>
      <span class="masthead-kicker">Analyse spatiale</span>
      <h1>Carte de susceptibilité<br>aux inondations</h1>
    </div>
    <div class="masthead-meta">
      <strong>Modèle</strong>
      Stacking — RF + SVM + XGB<br>
      AUC 0.927 · Test équilibré<br>
      <strong style="margin-top:1rem;">Zone</strong>
      Kosi Megafan · ~15 000 km²
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SECTION 01 — PERFORMANCE
# ============================================================
st.markdown("""
<div class="section-kicker">
  <span class="section-num">§ 01</span>
  <h2 class="section-title">Performance du modèle</h2>
</div>
""", unsafe_allow_html=True)

t = MODEL_METRICS["test"]
st.markdown(f'<div class="perf-grid">'
            f'<div class="perf-cell"><div class="perf-label">AUC-ROC</div><div class="perf-value">{t["AUC-ROC"]:.3f}</div></div>'
            f'<div class="perf-cell"><div class="perf-label">Accuracy</div><div class="perf-value">{t["Accuracy"]*100:.1f}%</div></div>'
            f'<div class="perf-cell"><div class="perf-label">Precision</div><div class="perf-value">{t["Precision"]*100:.1f}%</div></div>'
            f'<div class="perf-cell"><div class="perf-label">Recall</div><div class="perf-value">{t["Recall"]*100:.1f}%</div></div>'
            f'<div class="perf-cell"><div class="perf-label">F1-Score</div><div class="perf-value">{t["F1"]*100:.1f}%</div></div>'
            f'</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 02 — ARCHITECTURE
# ============================================================
st.markdown("""
<div class="section-kicker" style="margin-top:3rem;">
  <span class="section-num">§ 02</span>
  <h2 class="section-title">Architecture du modèle</h2>
</div>
""", unsafe_allow_html=True)

pipe_parts = []
for num, name, detail, sub in STACKING_COMPONENTS:
    pipe_parts.append(
        f'<div class="pipe-step"><span class="pipe-num">{num}</span>'
        f'<span class="pipe-name">{name}</span>'
        f'<span class="pipe-detail">{detail}<br>{sub}</span></div>'
    )
st.markdown(f'<div class="pipeline">{"".join(pipe_parts)}</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 03 — RÉPARTITION SPATIALE DU RISQUE
# ============================================================
st.markdown("""
<div class="section-kicker" style="margin-top:3rem;">
  <span class="section-num">§ 03</span>
  <h2 class="section-title">Répartition spatiale du risque</h2>
  <span class="section-note">Surface classifiée par niveau</span>
</div>
""", unsafe_allow_html=True)

if dist is not None:
    CLASS_COLORS = {1: "#7BA690", 2: "#C9A961", 3: "#C9704A", 4: "#8B2E2E"}
    CLASS_NAMES = {1: "Très faible", 2: "Faible", 3: "Élevé", 4: "Très élevé"}

    cells = []
    for cls in [1, 2, 3, 4]:
        n = dist["counts"][cls]
        area = dist["areas"][cls]
        pct = 100 * n / dist["total"] if dist["total"] else 0
        cells.append(
            f'<div class="risk-dist-cell" style="border-top:3px solid {CLASS_COLORS[cls]};">'
            f'<div class="risk-dist-label">'
            f'<span class="risk-dist-dot" style="background:{CLASS_COLORS[cls]};"></span>'
            f'{CLASS_NAMES[cls]}'
            f'</div>'
            f'<div class="risk-dist-count">{n:,}</div>'
            f'<div class="risk-dist-unit">pixels · {area:,.0f} km²</div>'
            f'<div class="risk-dist-pct">{pct:.1f}%</div>'
            f'</div>'
        )

    total_html = (
        f'<div class="risk-dist-total">'
        f'<span class="risk-dist-total-label">Surface totale analysée</span>'
        f'<span class="risk-dist-total-value">{dist["total"]:,} pixels · {dist["total_area"]:,.0f} km²</span>'
        f'</div>'
    )

    st.markdown(f'<div class="risk-distribution">{"".join(cells)}</div>{total_html}', unsafe_allow_html=True)
else:
    st.markdown('<div class="panel" style="color:var(--muted);font-style:italic;">Carte classifiée indisponible.</div>', unsafe_allow_html=True)

# ============================================================
# ONGLETS
# ============================================================
tab_map, tab_csv, tab_manual = st.tabs([
    "01 — Carte interactive",
    "02 — Import CSV",
    "03 — Saisie manuelle",
])

# ------------------------------------------------------------
# TAB 1 — CARTE
# ------------------------------------------------------------
with tab_map:
    try:
        overlay_uri, overlay_bounds_wgs, class_tif_used = generate_colored_overlay()
    except Exception as e:
        overlay_uri, overlay_bounds_wgs, class_tif_used = None, None, None
        st.warning(f"Couche colorée indisponible : {e}")

    if cm is not None:
        map_bounds_wgs = transform_bounds(cm["crs"], "EPSG:4326", *cm["bounds"])
    else:
        map_bounds_wgs = transform_bounds(crs, "EPSG:4326", *ref_bounds)
    center_lat = (map_bounds_wgs[1] + map_bounds_wgs[3]) / 2
    center_lon = (map_bounds_wgs[0] + map_bounds_wgs[2]) / 2

    left, right = st.columns([1.65, .75], gap="large")

    with left:
        st.markdown("""
        <div style="margin-bottom:1rem;">
          <div style="font-size:.66rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">Fig. 01</div>
          <h3 style="font-family:Fraunces,serif;font-size:1.35rem;font-weight:500;margin:0;color:var(--ink);">Vue d'ensemble du Kosi Megafan</h3>
        </div>
        """, unsafe_allow_html=True)

        m_folium = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                              tiles="OpenStreetMap", control_scale=True)
        if show_overlay and overlay_uri is not None:
            l, b, r, t_ = overlay_bounds_wgs
            folium.raster_layers.ImageOverlay(
                image=overlay_uri, bounds=[[b, l], [t_, r]],
                opacity=opacity_val, name="Susceptibilité", interactive=False,
            ).add_to(m_folium)
        folium.Rectangle(
            bounds=[[map_bounds_wgs[1], map_bounds_wgs[0]],
                    [map_bounds_wgs[3], map_bounds_wgs[2]]],
            color="#1A2B24", fill=False, weight=1.5,
        ).add_to(m_folium)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:#FDFBF8;padding:14px 16px;
                    border:1px solid #C9C0B0;font-family:'Inter',sans-serif;font-size:12px;">
          <div style="font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#1A2B24;font-size:10px;margin-bottom:8px;">Niveau de risque</div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><span style="display:inline-block;width:24px;height:3px;background:linear-gradient(90deg,#A8C9B5,#7BA690);"></span><span style="color:#2D3D35;">Très faible</span></div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><span style="display:inline-block;width:24px;height:3px;background:linear-gradient(90deg,#DFCB8A,#C9A961);"></span><span style="color:#2D3D35;">Faible</span></div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><span style="display:inline-block;width:24px;height:3px;background:linear-gradient(90deg,#D88A6A,#C9704A);"></span><span style="color:#2D3D35;">Élevé</span></div>
          <div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><span style="display:inline-block;width:24px;height:3px;background:linear-gradient(90deg,#A84A4A,#8B2E2E);"></span><span style="color:#2D3D35;">Très élevé</span></div>
        </div>
        """
        m_folium.get_root().html.add_child(folium.Element(legend_html))

        try:
            map_data = st_folium(m_folium, height=580, use_container_width=True, key="map")
        except TypeError:
            map_data = st_folium(m_folium, height=580, width=1000, key="map")

    with right:
        if cm is None:
            st.markdown('<div class="risk-card"><span class="risk-card-kicker">Indisponible</span><div class="risk-card-loc">Carte classifiée absente</div></div>', unsafe_allow_html=True)
        else:
            lat, lon = center_lat, center_lon
            if map_data and map_data.get("last_clicked"):
                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]

            location_name = reverse_geocode(lat, lon) or "Localisation non répertoriée"
            x_arr, y_arr = warp_transform("EPSG:4326", cm["crs"], [lon], [lat])
            x, y = x_arr[0], y_arr[0]
            b = cm["bounds"]

            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                st.markdown('<div class="risk-card"><span class="risk-card-kicker">Hors zone</span><div class="risk-card-loc">Point extérieur au domaine</div></div>', unsafe_allow_html=True)
            else:
                row, col = rowcol(cm["transform"], x, y)
                if row < 0 or row >= cm["shape"][0] or col < 0 or col >= cm["shape"][1]:
                    st.markdown('<div class="risk-card"><span class="risk-card-kicker">Hors grille</span><div class="risk-card-loc">Pixel invalide</div></div>', unsafe_allow_html=True)
                else:
                    displayed_class = int(cm["arr"][row, col])
                    feats, missing = sample_aligned_at_pixel(row, col, cm, srcs)
                    prob, _ = predict_single(feats, stacking, scaler, feature_names, ranks)
                    label = risk_level(prob)
                    displayed_label = CLASS_LABELS.get(displayed_class, "Hors zone")
                    risk_to_class = {"Très faible": 1, "Faible": 2, "Élevé": 3, "Très élevé": 4}
                    predicted_class = risk_to_class[label]
                    consistent = (predicted_class == displayed_class)
                    pct_marker = prob * 100

                    st.markdown(
                        f'<div class="risk-card">'
                        f'<span class="risk-card-kicker">Point sélectionné</span>'
                        f'<div class="risk-card-loc">{location_name}</div>'
                        f'<div class="risk-card-coords">{lat:.4f}° N · {lon:.4f}° E</div>'
                        f'<div class="risk-label-row"><span class="risk-level-name">{label}</span>'
                        f'<span class="risk-num">{prob*100:.0f}<span style="font-size:1.4rem;color:#A8B5AE;">%</span></span></div>'
                        f'<div class="spectrum"><div class="spectrum-track"></div>'
                        f'<div class="spectrum-marker" style="left:{pct_marker}%;"></div>'
                        f'<div class="spectrum-labels"><span>Faible</span><span>Élevé</span></div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    if consistent:
                        st.markdown(f'<div class="consistency-ok">Cohérent avec la carte — classe {displayed_class} ({displayed_label})</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="consistency-warn">Classe visuelle {displayed_class} ({displayed_label}) vs prédiction {predicted_class} ({label})</div>', unsafe_allow_html=True)

                    st.markdown('<div style="margin-top:1.8rem;font-size:.66rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:.7rem;">Facteurs dominants</div>', unsafe_allow_html=True)
                    for name, width in [("Altitude", 92), ("Végétation", 78), ("Densité de drainage", 70), ("Humidité du sol", 58), ("Pente", 43)]:
                        st.markdown(f'<div class="feature-line"><span class="feature-name">{name}</span><span class="feature-val">{width}%</span><div class="feature-bar"><span style="width:{width}%"></span></div></div>', unsafe_allow_html=True)

                    with st.expander("Valeurs détaillées"):
                        friendly_df = pd.DataFrame({
                            "Caractéristique": [FRIENDLY_NAMES[f] for f in RAW_FEATURES],
                            "Valeur": [f"{feats[f]:.2f} {UNITS.get(f, '')}".strip() for f in RAW_FEATURES],
                        })
                        st.dataframe(friendly_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 2 — CSV
# ------------------------------------------------------------
with tab_csv:
    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">Analyse par lot</div>
      <h3 class="panel-title">Classification d'un fichier CSV</h3>
    </div>
    """, unsafe_allow_html=True)

    col_tpl, col_up = st.columns([1, 1], gap="large")
    with col_tpl:
        st.markdown('<div class="panel"><div class="panel-kicker">Étape 01</div><h3 class="panel-title" style="font-size:1.1rem;">Template CSV</h3></div>', unsafe_allow_html=True)
        template = pd.DataFrame({f: [SLIDER_RANGES[f][2]] for f in RAW_FEATURES})
        st.download_button(
            "Télécharger le template",
            template.to_csv(index=False).encode("utf-8"),
            "template_facteurs.csv", "text/csv",
            use_container_width=True,
        )

    with col_up:
        st.markdown('<div class="panel"><div class="panel-kicker">Étape 02</div><h3 class="panel-title" style="font-size:1.1rem;">Import fichier</h3></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choisir un fichier CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing_cols = [c for c in RAW_FEATURES if c not in df_in.columns]
            if missing_cols:
                st.error(f"Colonnes manquantes : {missing_cols}")
            else:
                with st.spinner("Classification en cours…"):
                    probs = predict_batch(df_in, stacking, scaler, feature_names, ranks)

                df_out = df_in.copy()
                df_out["probability"] = probs.round(4)
                df_out["risk_class"] = [risk_level(p) for p in probs]

                st.markdown('<div style="margin:1.5rem 0 .7rem;font-size:.66rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);">Répartition</div>', unsafe_allow_html=True)

                counts = {lvl: int(sum(1 for p in probs if risk_level(p) == lvl))
                          for lvl in ["Très faible", "Faible", "Élevé", "Très élevé"]}
                total = len(probs)
                colors = ["#7BA690", "#C9A961", "#C9704A", "#8B2E2E"]
                grid_parts = []
                for lvl, color in zip(["Très faible", "Faible", "Élevé", "Très élevé"], colors):
                    pct = 100 * counts[lvl] / total if total else 0
                    grid_parts.append(
                        f'<div class="mini-kpi-cell" style="border-left:3px solid {color};">'
                        f'<span class="mini-kpi-lbl">{lvl}</span>'
                        f'<span class="mini-kpi-num">{counts[lvl]}</span>'
                        f'<span class="mini-kpi-pct">{pct:.1f}%</span></div>'
                    )
                st.markdown(f'<div class="mini-kpi-grid">{"".join(grid_parts)}</div>', unsafe_allow_html=True)

                st.markdown('<div style="margin:1.5rem 0 .7rem;font-size:.66rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);">Résultats</div>', unsafe_allow_html=True)
                display_df = df_out.copy()
                rename_map = {k: FRIENDLY_NAMES[k] for k in RAW_FEATURES}
                rename_map["probability"] = "Probabilité"
                rename_map["risk_class"]  = "Niveau de risque"
                display_df = display_df.rename(columns=rename_map)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "Télécharger les résultats",
                    df_out.to_csv(index=False).encode("utf-8"),
                    "predictions_flood.csv", "text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Erreur : {e}")

# ------------------------------------------------------------
# TAB 3 — SAISIE MANUELLE
# ------------------------------------------------------------
with tab_manual:
    st.markdown("""
    <div class="panel">
      <div class="panel-kicker">Simulation</div>
      <h3 class="panel-title">Estimation d'un point</h3>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1.35, .65], gap="large")

    with col_l:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-kicker">Topographie</div>', unsafe_allow_html=True)
        vals_topo = {}
        cols_topo = st.columns(2)
        for i, key in enumerate(["DEM", "Slope", "Aspect", "Curvature", "TWI"]):
            mn, mx, dv, stp = SLIDER_RANGES[key]
            unit = f" ({UNITS[key]})" if UNITS[key] else ""
            with cols_topo[i % 2]:
                vals_topo[key] = st.slider(f"{FRIENDLY_NAMES[key]}{unit}",
                    float(mn), float(mx), float(dv), float(stp),
                    help=DESCRIPTIONS[key], key=f"topo_{key}")

        st.markdown('<div class="panel-kicker" style="margin-top:1.5rem;">Occupation du sol & hydrologie</div>', unsafe_allow_html=True)
        vals_land = {}
        cols_land = st.columns(2)
        for i, key in enumerate(["NDVI", "NDBI", "DD", "DR", "GFI"]):
            mn, mx, dv, stp = SLIDER_RANGES[key]
            unit = f" ({UNITS[key]})" if UNITS[key] else ""
            with cols_land[i % 2]:
                vals_land[key] = st.slider(f"{FRIENDLY_NAMES[key]}{unit}",
                    float(mn), float(mx), float(dv), float(stp),
                    help=DESCRIPTIONS[key], key=f"land_{key}")
        st.markdown('</div>', unsafe_allow_html=True)

        feats_manual = {**vals_topo, **vals_land}

        clicked_manual = st.button(
            "▸  CALCULER LE RISQUE",
            type="primary",
            use_container_width=True,
            key="btn_manual",
        )

        if st.session_state.manual_result is not None:
            reset = st.button("Effacer", type="secondary", use_container_width=True, key="btn_reset")
            if reset:
                st.session_state.manual_result = None
                st.session_state.manual_snapshot = None
                st.session_state.manual_feats = None
                st.rerun()

    with col_r:
        if clicked_manual:
            with st.spinner("Calcul en cours…"):
                prob, _ = predict_single(feats_manual, stacking, scaler, feature_names, ranks)
            st.session_state.manual_result = prob
            st.session_state.manual_snapshot = dict(feats_manual)
            st.session_state.manual_feats = dict(feats_manual)

        if st.session_state.manual_result is not None:
            prob = st.session_state.manual_result
            label = risk_level(prob)
            pct_marker = prob * 100

            snapshot = st.session_state.manual_snapshot or {}
            params_changed = any(
                abs(snapshot.get(k, 0) - feats_manual.get(k, 0)) > 1e-6
                for k in RAW_FEATURES
            )

            st.markdown(
                f'<div class="risk-card">'
                f'<span class="risk-card-kicker">Résultat</span>'
                f'<div class="risk-card-loc">{label}</div>'
                f'<div class="risk-card-coords">Paramètres personnalisés</div>'
                f'<div class="risk-label-row"><span class="risk-level-name">Probabilité</span>'
                f'<span class="risk-num">{prob*100:.0f}<span style="font-size:1.4rem;color:#A8B5AE;">%</span></span></div>'
                f'<div class="spectrum"><div class="spectrum-track"></div>'
                f'<div class="spectrum-marker" style="left:{pct_marker}%;"></div>'
                f'<div class="spectrum-labels"><span>Faible</span><span>Élevé</span></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            if params_changed:
                st.markdown(
                    '<div class="consistency-stale">'
                    'Paramètres modifiés — cliquez à nouveau sur <strong>Calculer le risque</strong>.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Valeurs utilisées"):
                feats_snap = st.session_state.manual_feats or feats_manual
                df_snap = pd.DataFrame({
                    "Caractéristique": [FRIENDLY_NAMES[f] for f in RAW_FEATURES],
                    "Valeur": [f"{feats_snap[f]:.2f} {UNITS.get(f, '')}".strip() for f in RAW_FEATURES],
                })
                st.dataframe(df_snap, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="risk-card" style="background:#FDFBF8;color:var(--ink);border:1px solid var(--line);">
              <span class="risk-card-kicker" style="color:var(--muted);">Simulation</span>
              <div class="risk-card-loc" style="color:var(--ink);font-size:1.1rem;">En attente</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--line);font-size:.72rem;color:var(--muted);font-style:italic;text-align:center;letter-spacing:.08em;">Kosi Atlas · Stacking V2 · Random Forest + SVM + XGBoost</div>', unsafe_allow_html=True)