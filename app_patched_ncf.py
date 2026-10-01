"""
Personalized Movie Recommendation System — patched Streamlit frontend

Compatible with the attached MovieLens + NCF notebook.

Expected files beside this app.py:
    app_patched_ncf.py
    ncf_model.pth
    recommendation_data.pkl

Run:
    streamlit run app_patched_ncf.py

Patch:
    - Keeps the exact NCF architecture used by the notebook.
    - Fixes new-user recommendation saturation.
    - Does NOT rank raw NCF predictions directly.
    - Builds a new-user vector from the movie embeddings of the movies
      the user rated.
    - Uses the user's learned genre affinity as a strong reranking signal.
    - Uses popularity only as a small fallback signal.
    - Always excludes movies already rated.
    - Cold-start quick-start movie cards can be shuffled/reloaded.
    - TMDB posters are shown throughout the catalog, rating screens,
      selected-movie search UI, and recommendation UI.

UI restyle (visual only — zero logic changes):
    - Dark cinematic theme (charcoal + ember red/amber, no blue/purple).
    - Outfit/Inter typography, hover-lift poster cards, rank + match
      badges, traffic-light rating chips, styled tabs/buttons/inputs.
    - Every widget key, session-state key, and code path is unchanged;
      drop-in replacement for the previous file.
"""

from html import escape as _esc
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "ncf_model.pth"
DATA_PATH = APP_DIR / "recommendation_data.pkl"

MIN_NEW_USER_RATINGS = 5
RATING_OPTS = ["–"] + [f"{x:.1f}" for x in np.arange(0.5, 5.01, 0.5)]

# ---------------------------------------------------------------------
# TMDB API
# ---------------------------------------------------------------------
# Paste your TMDB API key between the quotation marks below.
# Example:
# TMDB_API_KEY = "abc123..."
#
# This API is used ONLY to retrieve movie posters for the Streamlit UI.
# It does not participate in the recommendation model.
TMDB_API_KEY = "77afa0edcbdb43b83912ae0def60e2f4"

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"
TMDB_REQUEST_TIMEOUT = 3
POSTER_WORKERS = 8

st.set_page_config(
    page_title="Personalized Movie Recommendation",
    page_icon="🎬",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Dark cinematic theme — presentation layer only (no logic changes)
# ----------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#0d0f13; --bg-soft:#101318; --surface:#14171e; --surface-2:#1a1f27;
  --line:#232833; --line-soft:#1f242e;
  --ink:#f2ede3; --ink-dim:#b3ada1; --ink-faint:#837e74;
  --ember:#e8402f; --ember-hi:#f6573f; --amber:#e2a83d;
  --green:#5fbd8f; --red:#e07862;
}

*,*::before,*::after{ box-sizing:border-box; }

/* ---------- shell ---------- */
.stApp{
  background:var(--bg); color:var(--ink);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
}
[data-testid="stAppViewContainer"]{ background:var(--bg) !important; }
[data-testid="stAppViewContainer"]>.main{ background:var(--bg) !important; }
[data-testid="stHeader"]{ background:rgba(13,15,19,.86) !important; backdrop-filter:blur(10px); }
[data-testid="stSidebar"]{
  background:var(--bg-soft) !important;
  border-right:1px solid var(--line-soft);
}
[data-testid="stSidebar"] .block-container{ padding-top:1.8rem; }
[data-testid="stMainBlockContainer"], .block-container{
  padding-top:1.3rem !important; padding-bottom:4rem !important; max-width:none !important;
}
footer{ visibility:hidden; }
::selection{ background:rgba(232,64,47,.4); color:#fff; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:#272d38; border-radius:999px; border:2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover{ background:#333b48; }
.stApp a{ color:var(--ember-hi); }

h1,h2,h3,h4{ font-family:'Outfit','Inter',sans-serif; letter-spacing:-.015em; }
[data-testid="stMarkdownContainer"]{ color:#c6c0b4; }
[data-testid="stMarkdownContainer"] p{ color:#c6c0b4; line-height:1.6; }
[data-testid="stMarkdownContainer"] strong{ color:var(--ink); }
[data-testid="stMarkdownContainer"] hr, hr{ border-color:var(--line) !important; }

/* ---------- brand (sidebar) ---------- */
.brand{ display:flex; align-items:center; gap:12px; padding:2px 0 6px; }
.brand-badge{
  flex:0 0 auto; width:38px; height:38px; border-radius:11px;
  background:linear-gradient(145deg,var(--ember-hi),#c22f21);
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 8px 22px rgba(232,64,47,.35), inset 0 1px 0 rgba(255,255,255,.18);
}
.brand-badge svg{ width:16px; height:16px; fill:#fff; margin-left:2px; }
.brand-name{ font:700 16px/1.15 'Outfit',sans-serif; color:var(--ink); }
.brand-sub{
  font:500 9.5px/1.35 'Inter',sans-serif; color:var(--ink-faint);
  text-transform:uppercase; letter-spacing:.09em; margin-top:3px;
}

/* ---------- hero ---------- */
.hero{ position:relative; padding:2.1rem 0 1.5rem; margin-bottom:.2rem; }
.hero::before{
  content:""; position:absolute; left:-2.4rem; right:-2.4rem; top:-1.3rem; height:300px;
  background:
    radial-gradient(560px 210px at 88% -18%, rgba(232,64,47,.13), transparent 62%),
    radial-gradient(420px 190px at 6% 0%, rgba(226,168,61,.05), transparent 60%);
  pointer-events:none;
}
.hero-kicker{
  position:relative; font:600 10.5px 'Inter',sans-serif; letter-spacing:.24em;
  text-transform:uppercase; color:var(--ember); margin-bottom:10px;
}
.hero h1{
  position:relative; font:800 clamp(1.75rem,3.2vw,2.45rem)/1.08 'Outfit',sans-serif;
  margin:0 0 10px; color:var(--ink);
}
.hero h1 em{ font-style:normal; color:var(--ember-hi); }
.hero p{ position:relative; margin:0 0 16px; max-width:680px; font:400 .92rem/1.65 'Inter',sans-serif; color:#a49e92; }
.hero-stats{ position:relative; display:flex; gap:8px; flex-wrap:wrap; }
.stat-chip{
  display:inline-flex; align-items:baseline; gap:6px; padding:6px 13px;
  border-radius:999px; border:1px solid var(--line); background:var(--surface);
  font:500 12px 'Inter',sans-serif; color:var(--ink-dim);
}
.stat-chip b{ font:700 13px 'Outfit',sans-serif; color:var(--ink); }

/* ---------- section headers ---------- */
.sec{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin:1.7rem 0 1rem; }
.sec-kicker{
  font:600 10px 'Inter',sans-serif; letter-spacing:.2em;
  text-transform:uppercase; color:var(--ember); margin-bottom:5px;
}
.sec-title{
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  font:700 1.3rem/1.15 'Outfit',sans-serif; color:var(--ink); letter-spacing:-.01em;
}
.sec-sub{ margin-top:6px; font:400 .84rem/1.5 'Inter',sans-serif; color:var(--ink-faint); }
.sec-chip{
  display:inline-flex; align-items:center; padding:4px 11px; border-radius:999px;
  border:1px solid var(--line); background:var(--surface);
  font:600 11.5px 'Inter',sans-serif; color:var(--ink-dim);
}

/* ---------- movie grid & cards ---------- */
.mv-grid{
  display:grid; grid-template-columns:repeat(auto-fill,minmax(158px,1fr));
  gap:22px 16px; margin:.4rem 0 1rem;
}
.mv-card{ min-width:0; }
.mv-poster{
  position:relative; aspect-ratio:2/3; border-radius:12px; overflow:hidden;
  background:var(--surface-2); border:1px solid rgba(255,255,255,.06);
  box-shadow:0 2px 10px rgba(0,0,0,.28);
  transition:transform .24s cubic-bezier(.22,.68,.32,1), box-shadow .24s, border-color .24s;
}
.mv-poster img{ width:100%; height:100%; object-fit:cover; display:block; transition:transform .4s ease; }
.mv-card:hover .mv-poster{
  transform:translateY(-6px); border-color:rgba(232,64,47,.4);
  box-shadow:0 22px 44px rgba(0,0,0,.55), 0 0 0 1px rgba(232,64,47,.22);
}
.mv-card:hover .mv-poster img{ transform:scale(1.06); }
.mv-poster.solo{ box-shadow:0 8px 26px rgba(0,0,0,.4); }
.mv-ph{
  position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
  justify-content:center; gap:10px; color:#5b564d;
  background:linear-gradient(165deg,#1c212b 0%,#12151c 70%);
}
.mv-ph svg{ width:34px; height:34px; stroke:#5b564d; }
.mv-ph span{ font:600 9.5px 'Inter',sans-serif; letter-spacing:.18em; text-transform:uppercase; }
.mv-rank{
  position:absolute; top:9px; left:9px; z-index:2; padding:3px 9px; border-radius:8px;
  background:var(--ember); color:#fff; font:700 12.5px 'Outfit',sans-serif;
  box-shadow:0 6px 16px rgba(232,64,47,.45);
}
.mv-match{
  position:absolute; top:9px; right:9px; z-index:2; padding:3px 8px; border-radius:8px;
  background:rgba(10,12,16,.78); backdrop-filter:blur(6px);
  border:1px solid rgba(255,255,255,.14);
  font:600 10.5px 'Inter',sans-serif; color:var(--green);
}
.mv-match.mid{ color:var(--amber); }
.mv-match.low{ color:var(--ink-faint); }
.mv-title{
  margin:10px 0 2px; min-height:2.7em; font:600 13px/1.35 'Outfit',sans-serif; color:var(--ink);
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}
.mv-genres{
  font:400 11px/1.4 'Inter',sans-serif; color:var(--ink-faint);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.mv-meta{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }
.star{ color:var(--amber); }
.chip{
  display:inline-flex; align-items:center; gap:4px; padding:3px 8px; border-radius:7px;
  background:var(--surface); border:1px solid var(--line);
  font:500 10.5px 'Inter',sans-serif; color:var(--ink-dim);
}
.chip.tone-good{ color:var(--green); border-color:rgba(95,189,143,.28); background:rgba(95,189,143,.07); }
.chip.tone-mid{ color:var(--amber); border-color:rgba(226,168,61,.28); background:rgba(226,168,61,.07); }
.chip.tone-low{ color:var(--red); border-color:rgba(224,120,98,.28); background:rgba(224,120,98,.07); }

/* quick-start mini card (inside the rating form) */
.qs-avg{ margin-top:6px; font:500 11px 'Inter',sans-serif; color:var(--ink-dim); }

/* detail view */
.detail-title{ font:700 1.35rem/1.2 'Outfit',sans-serif; color:var(--ink); margin-bottom:8px; }
.detail-genres{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }
.g-chip{
  padding:3px 10px; border-radius:999px; border:1px solid var(--line);
  background:var(--surface); font:500 11px 'Inter',sans-serif; color:var(--ink-dim);
}
.detail-meta{ display:flex; gap:6px; flex-wrap:wrap; }

/* learned-taste chips */
.taste{ display:flex; align-items:center; flex-wrap:wrap; gap:7px; margin:.2rem 0 1.1rem; }
.taste-label{
  font:600 10px 'Inter',sans-serif; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-faint); margin-right:3px;
}
.taste-chip{
  padding:4px 12px; border-radius:999px; background:rgba(226,168,61,.08);
  border:1px solid rgba(226,168,61,.26); font:500 12px 'Inter',sans-serif; color:#e8c07a;
}
.taste-chip b{ font:700 12px 'Outfit',sans-serif; color:#f0cf96; }

/* pager / counts / table titles */
.pager{ text-align:center; padding-top:5px; }
.page-chip{
  display:inline-flex; align-items:center; gap:5px; padding:6px 15px; border-radius:999px;
  border:1px solid var(--line); background:var(--surface);
  font:500 12px 'Inter',sans-serif; color:var(--ink-dim);
}
.page-chip b{ font:700 12.5px 'Outfit',sans-serif; color:var(--ink); }
.count-line{ margin:.2rem 0 .8rem; }
.table-title{ font:700 1.02rem 'Outfit',sans-serif; color:var(--ink); margin:1.3rem 0 .6rem; }
.tt-sub{ font:500 12px 'Inter',sans-serif; color:var(--ink-faint); }

/* ---------- tabs ---------- */
[data-testid="stTabs"]{ margin-top:.4rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:0 !important; border-bottom:1px solid var(--line) !important; width:100%; }
[data-testid="stTabs"] [data-baseweb="tab"]{
  background-color:transparent !important; border-radius:0 !important;
  padding:11px 4px !important; margin-right:26px !important;
  font:600 .93rem 'Outfit',sans-serif !important; color:var(--ink-faint) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{ color:#d9d3c7 !important; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"]{ color:var(--ink) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{
  background-color:var(--ember) !important; height:3px !important; border-radius:3px 3px 0 0;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{ padding-top:1.1rem; }

/* ---------- widget labels ---------- */
[data-testid="stWidgetLabel"] p{
  font:500 .8rem 'Inter',sans-serif !important; color:var(--ink-dim) !important; margin-bottom:.4rem;
}

/* ---------- buttons ---------- */
[data-testid="stButton"] > button, .stButton > button{
  background:var(--surface) !important; border:1px solid #2b3240 !important;
  color:#ddd7cb !important; border-radius:10px !important;
  font:600 .84rem 'Inter',sans-serif !important; padding:.42rem 1.05rem !important;
  transition:all .16s ease !important;
}
[data-testid="stButton"] > button:hover, .stButton > button:hover{
  border-color:rgba(232,64,47,.55) !important; background:var(--surface-2) !important;
  color:#fff !important; transform:translateY(-1px); box-shadow:0 8px 20px rgba(0,0,0,.35);
}
[data-testid="stButton"] > button:focus:not(:active), .stButton > button:focus:not(:active){
  border-color:var(--ember) !important; box-shadow:0 0 0 3px rgba(232,64,47,.2) !important;
}
[data-testid="stButton"] > button[kind="primary"],
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"]{
  background:var(--ember) !important; border-color:var(--ember) !important;
  color:#fff !important; box-shadow:0 6px 18px rgba(232,64,47,.28);
}
[data-testid="stButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover{
  background:var(--ember-hi) !important; border-color:var(--ember-hi) !important;
}
[data-testid="stFormSubmitButton"] > button{
  background:var(--ember) !important; border:1px solid var(--ember) !important;
  color:#fff !important; border-radius:10px !important;
  font:600 .86rem 'Inter',sans-serif !important;
  box-shadow:0 6px 18px rgba(232,64,47,.26); transition:all .16s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover{
  background:var(--ember-hi) !important; box-shadow:0 10px 26px rgba(232,64,47,.4) !important;
  transform:translateY(-1px);
}

/* ---------- text inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stTextInputField"]{
  background-color:var(--surface) !important; border:1px solid #2b3240 !important;
  border-radius:10px !important; color:var(--ink) !important;
  font:400 .88rem 'Inter',sans-serif !important;
}
[data-testid="stTextInputRootElement"]{
  background-color:transparent !important; border-color:transparent !important;
}
[data-testid="stTextInput"] input:focus{
  border-color:var(--ember) !important; box-shadow:0 0 0 3px rgba(232,64,47,.14) !important;
}
[data-testid="stTextInput"] input::placeholder{ color:#6c675e !important; }

/* ---------- select / multiselect ---------- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child,
[data-testid="stMultiselect"] [data-baseweb="select"] > div:first-child{
  background-color:var(--surface) !important; border:1px solid #2b3240 !important;
  border-radius:10px !important; color:var(--ink) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div{
  color:var(--ink) !important; font:400 .88rem 'Inter',sans-serif !important;
}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg{ fill:#8b8579 !important; }
[data-baseweb="popover"]{
  background-color:#161b23 !important; border:1px solid #2b3240 !important;
  border-radius:12px !important; box-shadow:0 24px 60px rgba(0,0,0,.55) !important;
}
[data-baseweb="popover"] ul{ background-color:transparent !important; padding:6px !important; }
[data-baseweb="popover"] li{ border-radius:8px !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li *{ color:#d8d2c6; }
[data-baseweb="popover"] li[aria-selected="true"]{ background:rgba(232,64,47,.16) !important; }
[data-baseweb="tag"]{
  background-color:rgba(232,64,47,.13) !important; border-radius:8px !important;
  font:500 .8rem 'Inter',sans-serif !important;
}
[data-baseweb="tag"] span{ color:#f0b6ac !important; }

/* ---------- radio (sidebar profile selector) ---------- */
[data-testid="stRadio"] label{
  display:flex; align-items:center; border:1px solid var(--line);
  background:var(--surface); border-radius:10px; padding:9px 12px;
  margin-bottom:8px; cursor:pointer; transition:all .15s ease; color:#d3cdc1;
}
[data-testid="stRadio"] label:hover{ border-color:#39414f; }
[data-testid="stRadio"] label:has(input:checked){
  border-color:rgba(232,64,47,.65); background:rgba(232,64,47,.07);
  box-shadow:0 0 0 1px rgba(232,64,47,.28);
}
[data-testid="stRadio"] [role="radiobutton"][aria-checked="true"]{ border-color:var(--ember) !important; }

/* ---------- slider ---------- */
[data-testid="stSlider"]{ color:#9a948a; }
[data-testid="stSlider"] [role="slider"]{
  background-color:var(--ember) !important; border:2px solid #14171e !important;
  box-shadow:0 0 0 4px rgba(232,64,47,.16), 0 3px 10px rgba(0,0,0,.4) !important;
}

/* ---------- metrics ---------- */
[data-testid="stMetric"]{
  background:var(--surface); border:1px solid var(--line);
  border-radius:13px; padding:15px 17px;
}
[data-testid="stMetricLabel"] p{
  font:600 10.5px 'Inter',sans-serif !important; text-transform:uppercase;
  letter-spacing:.09em; color:var(--ink-faint) !important;
}
[data-testid="stMetricValue"]{ font:700 1.85rem 'Outfit',sans-serif !important; color:var(--ink) !important; }

/* ---------- dataframe ---------- */
[data-testid="stDataFrame"]{
  border:1px solid var(--line) !important; border-radius:12px !important; overflow:hidden;
}

/* ---------- alerts ---------- */
[data-testid="stAlert"]{
  background-color:#181c22 !important; border:1px solid var(--line) !important;
  border-left:3px solid var(--amber) !important; border-radius:12px !important;
  color:#d9d3c7 !important;
}
[data-testid="stAlert"] *{ color:#d9d3c7 !important; }
[data-testid="stAlert"] svg{ color:var(--amber) !important; fill:var(--amber) !important; }

/* ---------- form ---------- */
[data-testid="stForm"]{
  border:1px solid var(--line) !important; border-radius:14px !important;
  background:#10141a !important; padding:1.3rem 1.3rem 1.1rem !important;
}

/* ---------- streamlit >= 1.5x react-aria widgets ---------- */
/* combobox control (selectbox + multiselect, closed state) */
.react-aria-ComboBox [data-rac][role="group"]{
  background-color:var(--surface) !important;
  border:1px solid #2b3240 !important; border-radius:10px !important;
}
.react-aria-ComboBox input{
  background-color:transparent !important; color:var(--ink) !important;
  font:400 .88rem 'Inter',sans-serif !important;
}
.react-aria-ComboBox input::placeholder{ color:#6c675e !important; }
.react-aria-ComboBox svg{ color:#8b8579 !important; }
.react-aria-ComboBox [data-rac]:focus-within{
  border-color:var(--ember) !important;
  box-shadow:0 0 0 3px rgba(232,64,47,.14) !important;
}

/* dropdown popover + options (portaled to body) */
div:has(> [role="listbox"]){
  background-color:#161b23 !important;
  border:1px solid #2b3240 !important; border-radius:12px !important;
  box-shadow:0 24px 60px rgba(0,0,0,.55) !important;
}
[role="listbox"]{ background-color:transparent !important; }
[role="option"]{ color:#d8d2c6 !important; border-radius:8px; }
[role="option"]:hover{ background-color:#1c212b !important; }
[role="option"][aria-selected="true"]{
  background-color:rgba(232,64,47,.16) !important; color:#fff !important;
}

/* multiselect tags */
[data-testid="stMultiSelectTagsContainer"] > *{
  background-color:rgba(232,64,47,.13) !important;
  border-radius:8px !important; color:#f0b6ac !important;
}
[data-testid="stMultiSelectTagsContainer"] *{ color:#f0b6ac !important; }
[data-testid="stMultiSelectTagsContainer"] svg{ color:#8b8579 !important; }

/* tabs (react-aria variants) */
[data-testid="stTabs"] [role="tablist"]{
  gap:0 !important; border-bottom:1px solid var(--line) !important; width:100%;
}
[data-testid="stTabs"] [data-testid="stTab"]{
  background-color:transparent !important; border-radius:0 !important;
  padding:11px 4px !important; margin-right:26px !important;
  font:600 .93rem 'Outfit',sans-serif !important; color:var(--ink-faint) !important;
}
[data-testid="stTab"] p{ color:inherit !important; font:inherit !important; margin:0 !important; }
[data-testid="stTabs"] [data-testid="stTab"]:hover{ color:#d9d3c7 !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
[data-testid="stTabs"] [data-testid="stTab"][data-selected="true"]{ color:var(--ink) !important; }
[data-testid="stTab"] .react-aria-SelectionIndicator{
  background-color:var(--ember) !important; height:3px !important;
  border-radius:3px 3px 0 0;
}

/* slider labels + value bubble */
[data-testid="stSliderTickBar"]{ color:var(--ink-faint) !important; }
[data-testid="stSliderThumbValue"]{ color:var(--ember-hi) !important; }

/* radio (react-aria attribute fallback) */
[data-testid="stRadioOption"]{
  border:1px solid var(--line); background:var(--surface);
  border-radius:10px; padding:9px 12px; margin-bottom:8px;
  cursor:pointer; transition:all .15s ease;
}
[data-testid="stRadioOption"]:hover{ border-color:#39414f; }
[data-testid="stRadioOption"][data-selected="true"]{
  border-color:rgba(232,64,47,.65); background:rgba(232,64,47,.07);
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Presentation helpers (pure UI — they only format already-computed data)
# ----------------------------------------------------------------------------
def _rating_tone(value):
    """Traffic-light tone for a 0.5-5 rating."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "low"
    if value >= 4.0:
        return "good"
    if value >= 2.5:
        return "mid"
    return "low"


def _match_tone(score):
    """Traffic-light tone for a 0-1 match score."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "low"
    if score >= 0.75:
        return "good"
    if score >= 0.50:
        return "mid"
    return "low"


def _fmt_count(value):
    """1234 -> 1.2K, 1250000 -> 1.3M (display only)."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "–"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value}"


def _clean_genres(genres):
    """Split a MovieLens 'A|B|C' genre string for display."""
    return [
        g
        for g in str(genres or "").split("|")
        if g and g != "(no genres listed)"
    ]


def _poster_inner_html(poster_url, title):
    """Poster <img> or a styled placeholder when no artwork exists."""
    if poster_url:
        return (
            f'<img src="{_esc(poster_url)}" alt="{_esc(title)}" loading="lazy">'
        )
    return (
        '<div class="mv-ph">'
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M4 4h16a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5'
        'a1 1 0 0 1 1-1zm3 0v16m10-16v16M3 9h4m10 0h4M3 15h4m10 0h4" '
        'stroke-width="1.6" stroke-linecap="round"/>'
        "</svg>"
        "<span>No artwork</span>"
        "</div>"
    )


def sec_header(kicker, title, chip=None, sub=None):
    """Styled section header (kicker + title + optional chip / subtitle)."""
    chip_html = f'<span class="sec-chip">{_esc(chip)}</span>' if chip else ""
    sub_html = f'<div class="sec-sub">{_esc(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="sec"><div class="sec-text">'
        f'<div class="sec-kicker">{_esc(kicker)}</div>'
        f'<div class="sec-title">{_esc(title)}{chip_html}</div>'
        f"{sub_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_hero(n_movies, n_users, n_ratings):
    """Compact hero band above the tabs (display only)."""
    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-kicker">Neural collaborative filtering · MovieLens</div>'
        f"<h1>Find your next <em>favorite film</em></h1>"
        f"<p>Rate a handful of movies or load an existing MovieLens profile. "
        f"The trained NCF model scores the full catalog, and a "
        f"preference-aware reranker blends your genre affinity into the "
        f"final ranking.</p>"
        f'<div class="hero-stats">'
        f'<span class="stat-chip"><b>{n_movies:,}</b> movies</span>'
        f'<span class="stat-chip"><b>{n_users:,}</b> users</span>'
        f'<span class="stat-chip"><b>{n_ratings:,}</b> ratings</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# NCF model — exact architecture used by the attached notebook
# ----------------------------------------------------------------------------
class NCF(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=16,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        self.user_bias = nn.Embedding(
            num_users,
            1
        )

        self.movie_bias = nn.Embedding(
            num_movies,
            1
        )

        self.emb_dropout = nn.Dropout(0.3)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(16, 1),
        )

        nn.init.normal_(
            self.user_embedding.weight,
            std=0.05,
        )

        nn.init.normal_(
            self.movie_embedding.weight,
            std=0.05,
        )

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

        nn.init.constant_(
            self.mlp[12].bias,
            0.69,
        )

    def forward(self, users, movies):

        user_vector = self.emb_dropout(
            self.user_embedding(users)
        )

        movie_vector = self.emb_dropout(
            self.movie_embedding(movies)
        )

        x = torch.cat(
            [user_vector, movie_vector],
            dim=1,
        )

        mlp_out = self.mlp(x).squeeze(-1)

        dot = (
            user_vector * movie_vector
        ).sum(dim=1)

        u_b = self.user_bias(
            users
        ).squeeze(-1)

        m_b = self.movie_bias(
            movies
        ).squeeze(-1)

        raw = (
            mlp_out
            + dot
            + u_b
            + m_b
        )

        return (
            0.5
            + 4.5 * torch.sigmoid(raw)
        )

    def score(
        self,
        user_vec,
        user_b,
        movies,
    ):
        movie_vec = self.movie_embedding(
            movies
        )

        if user_vec.dim() == 1:
            user_vec = user_vec.unsqueeze(0)

        user_vec = user_vec.expand(
            movie_vec.size(0),
            -1,
        )

        user_b = torch.as_tensor(
            user_b,
            dtype=movie_vec.dtype,
            device=movie_vec.device,
        )

        if user_b.dim() == 0:
            user_b = user_b.view(1)

        user_b = user_b.reshape(1).expand(
            movie_vec.size(0)
        )

        x = torch.cat(
            [user_vec, movie_vec],
            dim=1,
        )

        mlp_out = self.mlp(
            x
        ).squeeze(-1)

        dot = (
            user_vec * movie_vec
        ).sum(dim=1)

        raw = (
            mlp_out
            + dot
            + user_b
            + self.movie_bias(
                movies
            ).squeeze(-1)
        )

        return (
            0.5
            + 4.5 * torch.sigmoid(raw)
        )


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NCF model…")
def load_model(path: str):

    payload = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(payload, dict)
        and "model_state_dict" in payload
    ):
        state = payload["model_state_dict"]
    else:
        state = payload

    if not isinstance(state, dict):
        raise ValueError(
            "ncf_model.pth must contain "
            "a state_dict or a dictionary "
            "containing model_state_dict."
        )

    # Notebook uses:
    # num_users, num_movies, embedding_dim=16
    num_users, embedding_dim = (
        state["user_embedding.weight"].shape
    )

    num_movies = (
        state["movie_embedding.weight"].shape[0]
    )

    model = NCF(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=embedding_dim,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model


# ----------------------------------------------------------------------------
# Load notebook-generated recommendation data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading recommendation data…")
def load_data(path: str):

    with open(path, "rb") as f:
        data = pickle.load(f)

    required = {
        "user_to_idx",
        "movie_to_idx",
        "movies_df",
        "ratings_df",
    }

    missing = required - set(data.keys())

    if missing:
        raise ValueError(
            "recommendation_data.pkl is missing: "
            + ", ".join(sorted(missing))
        )

    ratings = data["ratings_df"].copy()
    movies = data["movies_df"].copy()

    user_to_idx = {
        int(k): int(v)
        for k, v in data["user_to_idx"].items()
    }

    movie_to_idx = {
        int(k): int(v)
        for k, v in data["movie_to_idx"].items()
    }

    ratings["userId"] = ratings[
        "userId"
    ].astype(int)

    ratings["movieId"] = ratings[
        "movieId"
    ].astype(int)

    ratings["rating"] = ratings[
        "rating"
    ].astype(float)

    movies["movieId"] = movies[
        "movieId"
    ].astype(int)

    ratings["user_idx"] = ratings[
        "userId"
    ].map(user_to_idx)

    ratings["movie_idx"] = ratings[
        "movieId"
    ].map(movie_to_idx)

    if (
        ratings["user_idx"].isna().any()
        or ratings["movie_idx"].isna().any()
    ):
        raise ValueError(
            "Stored user/movie mappings do not "
            "match ratings_df."
        )

    ratings["user_idx"] = ratings[
        "user_idx"
    ].astype(int)

    ratings["movie_idx"] = ratings[
        "movie_idx"
    ].astype(int)

    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(
            avg_rating="mean",
            n_ratings="count",
        )
        .reset_index()
    )

    catalog = movies.merge(
        stats,
        on="movieId",
        how="inner",
    )

    catalog["movie_idx"] = catalog[
        "movieId"
    ].map(movie_to_idx)

    catalog = catalog.dropna(
        subset=["movie_idx"]
    ).copy()

    catalog["movie_idx"] = catalog[
        "movie_idx"
    ].astype(int)

    catalog = catalog.sort_values(
        "movie_idx"
    ).reset_index(drop=True)

    # Smoothed popularity for cold start / reranking.
    global_mean = float(
        ratings["rating"].mean()
    )

    prior_count = 20

    n = catalog[
        "n_ratings"
    ].astype(float)

    catalog["popularity"] = (
        n / (n + prior_count)
        * catalog["avg_rating"]
        +
        prior_count
        / (n + prior_count)
        * global_mean
    )

    genres = sorted(
        {
            genre
            for genre_string in catalog[
                "genres"
            ].fillna("")
            for genre in str(
                genre_string
            ).split("|")
            if genre
            and genre != "(no genres listed)"
        }
    )

    return (
        ratings,
        catalog,
        movies,
        user_to_idx,
        movie_to_idx,
        genres,
        global_mean,
    )


# ----------------------------------------------------------------------------
# TMDB poster helper
# ----------------------------------------------------------------------------
def split_movie_title_year(title):
    """
    MovieLens titles commonly look like:
        Toy Story (1995)

    Return:
        clean title, year
    """
    import re

    title = str(title).strip()

    match = re.search(r"\((\d{4})\)\s*$", title)

    if match:
        year = int(match.group(1))
        clean_title = title[:match.start()].strip()
        return clean_title, year

    return title, None


@st.cache_data(show_spinner=False, ttl=7 * 24 * 60 * 60)
def get_tmdb_poster(title, api_key):
    """
    Search TMDB for a movie title and return its poster URL.

    Cached for 7 days so the same movie does not trigger another API
    search on every Streamlit rerun.
    """
    if not api_key:
        return None

    api_key = str(api_key).strip()

    if (
        not api_key
        or api_key == "PASTE_YOUR_TMDB_API_KEY_HERE"
    ):
        return None

    clean_title, year = split_movie_title_year(title)

    params = {
        "api_key": api_key,
        "query": clean_title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }

    if year is not None:
        params["year"] = year

    try:
        response = requests.get(
            TMDB_SEARCH_URL,
            params=params,
            timeout=TMDB_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            return None

        selected = None

        for result in results:
            result_title = str(result.get("title", "")).strip()
            release_date = str(result.get("release_date", ""))

            result_year = (
                int(release_date[:4])
                if len(release_date) >= 4
                and release_date[:4].isdigit()
                else None
            )

            if (
                result_title.lower() == clean_title.lower()
                and (year is None or result_year == year)
            ):
                selected = result
                break

        if selected is None:
            selected = results[0]

        poster_path = selected.get("poster_path")

        if not poster_path:
            return None

        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        return None


@st.cache_data(show_spinner=False, ttl=7 * 24 * 60 * 60)
def get_tmdb_posters(titles, api_key):
    """
    Resolve multiple posters concurrently.

    This avoids waiting for 20 sequential TMDB searches on a catalog page.
    """
    titles = tuple(dict.fromkeys(str(t) for t in titles))

    if not titles or not api_key or api_key == "PASTE_YOUR_TMDB_API_KEY_HERE":
        return {}

    results = {}

    # Multiple HTTP requests in parallel dramatically reduce initial page load
    # time while keeping the model itself completely unchanged.
    with ThreadPoolExecutor(max_workers=POSTER_WORKERS) as executor:
        futures = {
            executor.submit(get_tmdb_poster, title, api_key): title
            for title in titles
        }

        for future in as_completed(futures):
            title = futures[future]
            try:
                results[title] = future.result()
            except Exception:
                results[title] = None

    return results


def show_poster(
    title,
    api_key,
    width=170,
    poster_url=None,
):
    """Display a poster. Pass poster_url when a batch lookup was done."""
    if poster_url is None:
        poster_url = get_tmdb_poster(title, api_key)

    st.markdown(
        f'<div class="mv-poster solo" style="width:{int(width)}px;">'
        + _poster_inner_html(poster_url, title)
        + "</div>",
        unsafe_allow_html=True,
    )


def movie_card_html(
    row,
    rank=None,
    api_key="",
    user_rating=None,
    compact=False,
    poster_url=None,
):
    """Build the HTML for a single movie card (poster + badges + chips)."""
    title = str(
        row.get(
            "title",
            row.get("Title", ""),
        )
    )

    if poster_url is None:
        poster_url = get_tmdb_poster(title, api_key)

    poster = _poster_inner_html(poster_url, title)

    rank_badge = (
        f'<span class="mv-rank">{int(rank)}</span>'
        if rank is not None
        else ""
    )

    match_badge = ""

    predicted = row.get(
        "predicted_rating",
        np.nan,
    )

    match_score = row.get(
        "match_score",
        np.nan,
    )

    if pd.notna(predicted) and pd.notna(match_score):
        pct = int(round(float(match_score) * 100))
        match_badge = (
            f'<span class="mv-match {_match_tone(match_score)}">'
            f"{pct}% match</span>"
        )

    genre_list = _clean_genres(
        row.get(
            "genres",
            row.get("Genres", ""),
        )
    )

    title_clean, title_year = split_movie_title_year(title)

    genres_text = " · ".join(genre_list) if genre_list else "—"

    if title_year is not None:
        title_html = (
            f"{_esc(title_clean)} "
            '<span style="color:#837e74;font-weight:500;">'
            f"({_esc(str(title_year))})</span>"
        )
    else:
        title_html = _esc(title_clean)

    chips = []

    avg_rating = row.get("avg_rating", np.nan)
    n_ratings = row.get("n_ratings", np.nan)

    if pd.notna(avg_rating):
        chips.append(
            f'<span class="chip {_rating_tone(avg_rating)}">'
            f'<span class="star">★</span> {float(avg_rating):.1f}</span>'
        )

    if pd.notna(n_ratings):
        chips.append(
            f'<span class="chip">{_fmt_count(n_ratings)}</span>'
        )

    if pd.notna(predicted):
        chips.append(
            f'<span class="chip tone-mid">Est. {float(predicted):.1f}</span>'
        )

    if user_rating is not None and pd.notna(user_rating):
        chips.append(
            f'<span class="chip tone-good">You {float(user_rating):.1f}</span>'
        )

    if compact:
        chips = chips[:2]

    meta_html = "".join(chips)

    return (
        '<div class="mv-card">'
        f'<div class="mv-poster">{rank_badge}{match_badge}{poster}</div>'
        f'<div class="mv-title">{title_html}</div>'
        f'<div class="mv-genres" title="{_esc(genres_text)}">{_esc(genres_text)}</div>'
        f'<div class="mv-meta">{meta_html}</div>'
        "</div>"
    )


# ----------------------------------------------------------------------------
# Grid rendering (presentation only — data comes from the engine below)
# ----------------------------------------------------------------------------
def prefetch_posters(frame, api_key):
    """Batch-resolve TMDB posters for a dataframe of movies."""
    if frame is None or len(frame) == 0:
        return {}

    titles = frame["title"].astype(str).tolist()

    return get_tmdb_posters(titles, api_key)


def render_movie_grid(
    frame,
    api_key,
    ranked=False,
    user_ratings=None,
    poster_map=None,
    start_rank=1,
    compact=False,
):
    """Render a responsive grid of movie cards from a dataframe."""
    if frame is None or len(frame) == 0:
        st.markdown(
            '<div class="count-line">Nothing to show here yet.</div>',
            unsafe_allow_html=True,
        )
        return

    if poster_map is None:
        poster_map = prefetch_posters(frame, api_key)

    if user_ratings is None:
        user_ratings = {}

    cards = []

    for position, (_, row) in enumerate(frame.iterrows(), start=start_rank):
        rank = position if ranked else None

        cards.append(
            movie_card_html(
                row,
                rank=rank,
                api_key=api_key,
                user_rating=user_ratings.get(int(row["movieId"])),
                compact=compact,
                poster_url=poster_map.get(str(row["title"])),
            )
        )

    st.markdown(
        '<div class="mv-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Recommendation engine — the new-user patch
#
# Raw NCF predictions are NOT ranked directly (they saturate for new users).
# Instead:
#   1. A new-user vector is built from the movie embeddings of the movies
#      the user rated, weighted by their ratings.
#   2. The user's learned genre affinity is used as a strong reranking
#      signal.
#   3. Popularity is only a small fallback signal.
#   4. Movies already rated by the user are always excluded.
# ----------------------------------------------------------------------------
PRED_WEIGHT = 0.55
GENRE_WEIGHT = 0.35
POP_WEIGHT = 0.10


@st.cache_data(show_spinner=False)
def add_year_column(catalog: pd.DataFrame):
    """Derive a numeric release-year column from MovieLens titles."""
    framed = catalog.copy()

    framed["year"] = framed["title"].map(
        lambda t: split_movie_title_year(str(t))[1]
    )

    return framed


@st.cache_data(show_spinner=False)
def build_genre_lookup(catalog: pd.DataFrame):
    """movieId -> list of clean genre names (used for affinity scoring)."""
    return {
        int(movie_id): _clean_genres(genre_string)
        for movie_id, genre_string in zip(
            catalog["movieId"].astype(int),
            catalog["genres"].fillna(""),
        )
    }


@torch.no_grad()
def build_new_user_vector(model, user_ratings, movie_to_idx, global_mean):
    """
    Build a synthetic user representation from the movies just rated.

    The vector is the rating-weighted average of the movie embeddings
    the NCF model learned during training, so a brand-new user can be
    scored without ever having been seen by the network.

    Returns:
        (user_vector [embedding_dim] float32 tensor, user_bias float)
    """
    movie_ids = [
        int(movie_id)
        for movie_id in user_ratings.keys()
        if int(movie_id) in movie_to_idx
    ]

    if not movie_ids:
        return None, 0.0

    weights = np.array(
        [float(user_ratings[movie_id]) for movie_id in movie_ids],
        dtype=np.float32,
    )

    indices = torch.tensor(
        [movie_to_idx[movie_id] for movie_id in movie_ids],
        dtype=torch.long,
    )

    embeddings = model.movie_embedding.weight[indices].detach()

    weights_tensor = torch.from_numpy(weights).unsqueeze(1)

    user_vector = (
        (embeddings * weights_tensor).sum(dim=0)
        / weights_tensor.sum()
    )

    # Proxy bias: how far above/below the global mean this user rates.
    user_bias = float(weights.mean() - global_mean)

    return user_vector.cpu(), user_bias


@torch.no_grad()
def predict_with_user_vector(model, user_vector, user_bias, movie_indices, batch_size=8192):
    """Score candidate movies through the NCF network as a synthetic user."""
    predictions = []

    for start in range(0, len(movie_indices), batch_size):
        chunk = movie_indices[start : start + batch_size]

        chunk_tensor = torch.tensor(chunk, dtype=torch.long)

        chunk_scores = model.score(
            user_vector,
            user_bias,
            chunk_tensor,
        )

        predictions.append(chunk_scores.cpu().numpy())

    if not predictions:
        return np.array([])

    return np.concatenate(predictions)


@torch.no_grad()
def predict_with_user_index(model, user_idx, movie_indices, batch_size=8192):
    """Score candidate movies for an existing MovieLens user."""
    predictions = []

    for start in range(0, len(movie_indices), batch_size):
        chunk = movie_indices[start : start + batch_size]

        users = torch.full(
            (len(chunk),),
            int(user_idx),
            dtype=torch.long,
        )

        movies = torch.tensor(chunk, dtype=torch.long)

        chunk_scores = model(users, movies)

        predictions.append(chunk_scores.cpu().numpy())

    if not predictions:
        return np.array([])

    return np.concatenate(predictions)


def compute_genre_affinity(user_ratings, genre_lookup, global_mean, shrink=3.0):
    """
    Learned genre affinity from the user's own ratings.

    For every genre the user has touched, compute a centered, count-shrunk
    mean rating, then map it onto 0..1 (0.5 = neutral, > 0.5 liked,
    < 0.5 disliked). Shrinking keeps a single 5-star rating on one obscure
    movie from turning into a 100% affinity for its genre.
    """
    sums = {}
    counts = {}

    for movie_id, rating in user_ratings.items():
        for genre in genre_lookup.get(int(movie_id), []):
            sums[genre] = sums.get(genre, 0.0) + (float(rating) - global_mean)
            counts[genre] = counts.get(genre, 0) + 1

    affinity = {}

    for genre, total in sums.items():
        count = counts[genre]

        shrunk_mean = total / (count + shrink)

        affinity[genre] = float(np.clip(0.5 + shrunk_mean / 4.5, 0.0, 1.0))

    return affinity


def movie_genre_score(genre_string, affinity):
    """Mean affinity across a movie's genres (0.5 when unknown)."""
    genres = _clean_genres(genre_string)

    if not genres or not affinity:
        return 0.5

    values = [affinity.get(genre, 0.5) for genre in genres]

    return float(np.mean(values))


def rank_candidates(candidates, predictions, affinity):
    """
    Final ranking = NCF prediction + strong genre-affinity signal
    + small popularity fallback. Produces a 0..1 match_score for display.
    """
    ranked = candidates.copy()

    ranked["predicted_rating"] = predictions

    ranked["pred_norm"] = (ranked["predicted_rating"] - 0.5) / 4.5

    ranked["genre_score"] = ranked["genres"].fillna("").map(
        lambda genre_string: movie_genre_score(genre_string, affinity)
    )

    ranked["pop_norm"] = (ranked["popularity"] - 0.5) / 4.5

    ranked["final_score"] = (
        PRED_WEIGHT * ranked["pred_norm"]
        + GENRE_WEIGHT * ranked["genre_score"]
        + POP_WEIGHT * ranked["pop_norm"]
    )

    ranked["match_score"] = ranked["final_score"].clip(0.0, 1.0)

    return ranked.sort_values("final_score", ascending=False).reset_index(drop=True)


def recommend_for_new_user(
    model,
    catalog,
    movie_to_idx,
    genre_lookup,
    user_ratings,
    global_mean,
    top_n,
):
    """Top-N recommendations for a brand-new user of the app."""
    known_ids = [
        int(movie_id)
        for movie_id in user_ratings.keys()
        if int(movie_id) in movie_to_idx
    ]

    if len(known_ids) < MIN_NEW_USER_RATINGS:
        return None, None, 0, None

    user_vector, user_bias = build_new_user_vector(
        model,
        user_ratings,
        movie_to_idx,
        global_mean,
    )

    # Always exclude movies already rated.
    candidates = catalog.loc[
        ~catalog["movieId"].isin(list(user_ratings.keys()))
    ].copy()

    if candidates.empty:
        return None, None, 0, None

    movie_indices = candidates["movie_idx"].to_numpy(dtype=np.int64)

    predictions = predict_with_user_vector(
        model,
        user_vector,
        user_bias,
        movie_indices,
    )

    affinity = compute_genre_affinity(
        user_ratings,
        genre_lookup,
        global_mean,
    )

    ranked = rank_candidates(candidates, predictions, affinity)

    return ranked.head(top_n), affinity, len(candidates), None


def recommend_for_existing_user(
    model,
    all_ratings,
    catalog,
    user_id,
    user_to_idx,
    genre_lookup,
    global_mean,
    top_n,
):
    """Top-N recommendations for an existing MovieLens userId."""
    if user_id not in user_to_idx:
        return None, None, 0, None

    user_idx = user_to_idx[int(user_id)]

    user_history = all_ratings.loc[all_ratings["userId"] == int(user_id)]

    rated_ids = set(user_history["movieId"].astype(int).tolist())

    candidates = catalog.loc[~catalog["movieId"].isin(rated_ids)].copy()

    if candidates.empty:
        return None, None, 0, user_history

    movie_indices = candidates["movie_idx"].to_numpy(dtype=np.int64)

    predictions = predict_with_user_index(
        model,
        user_idx,
        movie_indices,
    )

    user_ratings_map = dict(
        zip(
            user_history["movieId"].astype(int),
            user_history["rating"].astype(float),
        )
    )

    affinity = compute_genre_affinity(
        user_ratings_map,
        genre_lookup,
        global_mean,
    )

    ranked = rank_candidates(candidates, predictions, affinity)

    return ranked.head(top_n), affinity, len(candidates), user_history


# ----------------------------------------------------------------------------
# UI — sidebar, pages, tabs & main
# ----------------------------------------------------------------------------
NEW_USER_LABEL = "🆕 New user — rate movies"
EXISTING_USER_LABEL = "🍿 Existing MovieLens user"

PAGE_SIZE = 12
QUICK_START_SIZE = 12

SORT_MODES = {
    "Popularity": ("popularity", False),
    "Average rating": ("avg_rating", False),
    "Number of ratings": ("n_ratings", False),
    "Title A–Z": ("title", True),
    "Newest first": ("year", False),
}


def _rerun():
    """Version-safe st.rerun()."""
    rerun = getattr(st, "rerun", None) or getattr(
        st, "experimental_rerun", None
    )

    if rerun is not None:
        rerun()


def render_brand():
    """Sidebar brand block."""
    st.sidebar.markdown(
        '<div class="brand">'
        '<div class="brand-badge">'
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M8 5v14l11-7z"/></svg>'
        "</div>"
        "<div>"
        '<div class="brand-name">CineMatch</div>'
        '<div class="brand-sub">MovieLens · NCF recommender</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def save_ratings_from_keys(prefix, movie_ids):
    """
    Merge selectbox values (keys like '<prefix>_<movieId>') from a submitted
    form into st.session_state['user_ratings'], then rerun.
    """
    updated = dict(st.session_state.get("user_ratings", {}))

    for movie_id in movie_ids:
        value = st.session_state.get(f"{prefix}_{int(movie_id)}", "–")

        if value != "–":
            updated[int(movie_id)] = float(value)
        else:
            updated.pop(int(movie_id), None)

    st.session_state["user_ratings"] = updated

    _rerun()


def _rating_index_for(movie_id, user_ratings):
    """Index into RATING_OPTS matching the user's saved rating (or '–')."""
    current = user_ratings.get(int(movie_id))

    if current is None:
        return 0

    label = f"{float(current):.1f}"

    if label not in RATING_OPTS:
        return 0

    return RATING_OPTS.index(label)


@st.cache_data(show_spinner=False)
def quick_start_pool(catalog: pd.DataFrame):
    """Popular, well-rated movies that make good first-rating prompts."""
    pool = catalog.loc[catalog["n_ratings"] >= 100]

    if len(pool) < QUICK_START_SIZE * 3:
        pool = catalog

    return pool.sort_values("popularity", ascending=False).head(400)


def render_quick_start(catalog, api_key, user_ratings):
    """Cold-start quick-start cards: rate a handful of famous movies."""
    sec_header(
        "Cold start",
        "Quick-start picks",
        chip="shuffle to refresh",
        sub=(
            "Rate whatever looks familiar — the recommendations wake up "
            f"after {MIN_NEW_USER_RATINGS} ratings."
        ),
    )

    seed = int(st.session_state.get("qs_seed", 0))

    pool = quick_start_pool(catalog)

    picks = pool.sample(
        n=min(QUICK_START_SIZE, len(pool)),
        random_state=seed,
    ).reset_index(drop=True)

    already_rated = int(
        picks["movieId"].astype(int).isin(list(user_ratings.keys())).sum()
    )

    qcol1, qcol2 = st.columns([1, 3])

    with qcol1:
        if st.button("🔀 Shuffle picks", key="qs_shuffle"):
            st.session_state["qs_seed"] = seed + 1
            _rerun()

    with qcol2:
        rated_note = (
            f" · {already_rated} already rated by you"
            if already_rated
            else ""
        )

        st.markdown(
            f'<div class="count-line">Showing {len(picks)} popular, '
            f"well-rated movies{rated_note}</div>",
            unsafe_allow_html=True,
        )

    poster_map = prefetch_posters(picks, api_key)

    with st.form("quick_start_form"):
        for start in range(0, len(picks), 6):
            chunk = picks.iloc[start : start + 6]

            cols = st.columns(6)

            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    st.markdown(
                        movie_card_html(
                            row,
                            api_key=api_key,
                            user_rating=user_ratings.get(int(row["movieId"])),
                            compact=True,
                            poster_url=poster_map.get(str(row["title"])),
                        ),
                        unsafe_allow_html=True,
                    )

                    st.selectbox(
                        "Your rating",
                        RATING_OPTS,
                        key=f"qs_{int(row['movieId'])}",
                        label_visibility="collapsed",
                        index=_rating_index_for(row["movieId"], user_ratings),
                    )

        submitted = st.form_submit_button(
            "Save quick-start ratings ⭐",
            use_container_width=True,
        )

    if submitted:
        save_ratings_from_keys(
            "qs",
            picks["movieId"].astype(int).tolist(),
        )


def render_search_and_rate(catalog, api_key, user_ratings):
    """Search any movie in the catalog and rate it."""
    sec_header(
        "Add more",
        "Search & rate any movie",
        sub="Simple substring search across the full catalog titles.",
    )

    query = st.text_input(
        "Movie title",
        key="rate_search",
        placeholder="Type part of a title — e.g. heat, amélie, dark knight…",
    )

    if not query.strip():
        return

    mask = catalog["title"].str.contains(
        query.strip(),
        case=False,
        regex=False,
        na=False,
    )

    results = catalog.loc[mask].head(12)

    st.markdown(
        f'<div class="count-line">{int(mask.sum()):,} matches · '
        f"showing first {len(results)}</div>",
        unsafe_allow_html=True,
    )

    if results.empty:
        st.info("Nothing matched that title — try a shorter fragment.")
        return

    poster_map = prefetch_posters(results, api_key)

    with st.form("rate_search_form"):
        for start in range(0, len(results), 4):
            chunk = results.iloc[start : start + 4]

            cols = st.columns(4)

            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    st.markdown(
                        movie_card_html(
                            row,
                            api_key=api_key,
                            user_rating=user_ratings.get(int(row["movieId"])),
                            poster_url=poster_map.get(str(row["title"])),
                        ),
                        unsafe_allow_html=True,
                    )

                    st.selectbox(
                        "Your rating",
                        RATING_OPTS,
                        key=f"rs_{int(row['movieId'])}",
                        label_visibility="collapsed",
                        index=_rating_index_for(row["movieId"], user_ratings),
                    )

        if st.form_submit_button(
            "Save these ratings",
            use_container_width=True,
        ):
            save_ratings_from_keys(
                "rs",
                results["movieId"].astype(int).tolist(),
            )


def render_your_ratings(catalog, user_ratings):
    """Everything the visitor has rated so far, plus progress to cold-start."""
    sec_header(
        "Your profile",
        f"You rated {len(user_ratings)} movies",
        chip=f"min {MIN_NEW_USER_RATINGS} for recommendations",
    )

    if not user_ratings:
        st.info(
            "No ratings yet — use the quick-start picks above or search "
            "for movies to rate."
        )
        return

    rated_ids = list(user_ratings.keys())

    rated = catalog.loc[catalog["movieId"].isin(rated_ids)].copy()

    rated["your_rating"] = rated["movieId"].map(
        lambda movie_id: user_ratings.get(int(movie_id))
    )

    rated = rated.sort_values("your_rating", ascending=False)

    table = rated[["title", "your_rating", "avg_rating", "n_ratings"]].copy()
    table.columns = ["Movie", "Your rating", "MovieLens avg", "# ratings"]

    st.dataframe(table, use_container_width=True)

    st.progress(min(len(user_ratings) / MIN_NEW_USER_RATINGS, 1.0))

    if len(user_ratings) < MIN_NEW_USER_RATINGS:
        st.warning(
            f"Rate at least {MIN_NEW_USER_RATINGS} movies to unlock "
            f"personalized recommendations "
            f"({MIN_NEW_USER_RATINGS - len(user_ratings)} to go)."
        )
    else:
        st.success(
            "You're all set — open the ✨ For you tab for your "
            "recommendations."
        )


def render_taste_chips(affinity, top_n=6):
    """Show the user's learned genre affinity as chips."""
    if not affinity:
        return

    top = sorted(affinity.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    html = (
        '<div class="taste">'
        '<span class="taste-label">Learned taste</span>'
    )

    for genre, score in top:
        html += (
            f'<span class="taste-chip">{_esc(genre)} '
            f"<b>{int(round(score * 100))}%</b></span>"
        )

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


def render_catalog(catalog, genres_list, api_key, user_ratings):
    """Full catalog browser: filter, sort, paginate, spotlight."""
    sec_header(
        "Catalog",
        "Browse every movie",
        chip=f"{len(catalog):,} titles",
    )

    fcol1, fcol2, fcol3 = st.columns([2, 1.4, 1.2])

    with fcol1:
        query = st.text_input(
            "Search",
            key="cat_search",
            placeholder="Search by title…",
        )

    with fcol2:
        genre_filter = st.multiselect(
            "Genres",
            genres_list,
            key="cat_genres",
        )

    with fcol3:
        sort_mode = st.selectbox(
            "Sort by",
            list(SORT_MODES.keys()),
            key="cat_sort",
        )

    min_avg = st.slider(
        "Minimum average rating",
        0.0,
        5.0,
        0.0,
        0.5,
        key="cat_min_avg",
    )

    filtered = catalog

    if query.strip():
        filtered = filtered.loc[
            filtered["title"].str.contains(
                query.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    if genre_filter:
        wanted = set(genre_filter)

        filtered = filtered.loc[
            filtered["genres"]
            .fillna("")
            .map(
                lambda genre_string: bool(
                    wanted.intersection(str(genre_string).split("|"))
                )
            )
        ]

    if min_avg > 0:
        filtered = filtered.loc[filtered["avg_rating"] >= min_avg]

    filtered = filtered.sort_values(
        SORT_MODES[sort_mode][0],
        ascending=SORT_MODES[sort_mode][1],
        na_position="last",
    )

    fingerprint = (
        query.strip().lower(),
        tuple(genre_filter),
        sort_mode,
        float(min_avg),
    )

    if st.session_state.get("cat_fp") != fingerprint:
        st.session_state["cat_fp"] = fingerprint
        st.session_state["cat_page"] = 0

    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)

    page = min(int(st.session_state.get("cat_page", 0)), total_pages - 1)

    if filtered.empty:
        st.markdown(
            '<div class="count-line">No movies match these filters.</div>',
            unsafe_allow_html=True,
        )
        st.info("Loosen the filters or clear the search box.")
        return

    start = page * PAGE_SIZE

    page_frame = filtered.iloc[start : start + PAGE_SIZE]

    st.markdown(
        f'<div class="count-line">Showing {start + 1}–{start + len(page_frame)} '
        f"of {len(filtered):,} movies · page {page + 1} / {total_pages}</div>",
        unsafe_allow_html=True,
    )

    poster_map = prefetch_posters(page_frame, api_key)

    render_movie_grid(
        page_frame,
        api_key,
        ranked=True,
        user_ratings=user_ratings,
        poster_map=poster_map,
        start_rank=start + 1,
    )

    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])

    with pcol1:
        if st.button(
            "← Previous",
            key="cat_prev",
            disabled=(page == 0),
            use_container_width=True,
        ):
            st.session_state["cat_page"] = page - 1
            _rerun()

    with pcol2:
        st.markdown(
            f'<div class="pager"><span class="page-chip">Page '
            f"<b>{page + 1}</b> of <b>{total_pages}</b></span></div>",
            unsafe_allow_html=True,
        )

    with pcol3:
        if st.button(
            "Next →",
            key="cat_next",
            disabled=(page >= total_pages - 1),
            use_container_width=True,
        ):
            st.session_state["cat_page"] = page + 1
            _rerun()

    sec_header(
        "Spotlight",
        "Inspect a movie",
        sub="Details for anything on the current page.",
    )

    titles = page_frame["title"].astype(str).tolist()

    choice = st.selectbox(
        "Pick a movie",
        titles,
        key="spotlight",
        label_visibility="collapsed",
    )

    spot = page_frame.loc[page_frame["title"].astype(str) == choice]

    if spot.empty:
        return

    row = spot.iloc[0]

    title_clean, title_year = split_movie_title_year(str(row["title"]))

    dcol1, dcol2 = st.columns([1, 3])

    with dcol1:
        show_poster(
            choice,
            api_key,
            width=200,
            poster_url=poster_map.get(choice),
        )

    with dcol2:
        year_html = (
            f' <span style="color:#837e74;font-weight:600;">'
            f"({_esc(str(title_year))})</span>"
            if title_year is not None
            else ""
        )

        genre_chips = "".join(
            f'<span class="g-chip">{_esc(genre)}</span>'
            for genre in _clean_genres(row.get("genres", ""))
        )

        meta_chips = []

        if pd.notna(row.get("avg_rating", np.nan)):
            meta_chips.append(
                f'<span class="chip {_rating_tone(row["avg_rating"])}">'
                f'<span class="star">★</span> {float(row["avg_rating"]):.1f}'
                "</span>"
            )

        if pd.notna(row.get("n_ratings", np.nan)):
            meta_chips.append(
                f'<span class="chip">{_fmt_count(row["n_ratings"])} ratings</span>'
            )

        if pd.notna(row.get("popularity", np.nan)):
            meta_chips.append(
                '<span class="chip tone-mid">Pop. '
                f'{float(row["popularity"]):.2f}</span>'
            )

        user_rating = user_ratings.get(int(row["movieId"]))

        if user_rating is not None:
            meta_chips.append(
                f'<span class="chip tone-good">You {float(user_rating):.1f}</span>'
            )

        st.markdown(
            f'<div class="detail-title">{_esc(title_clean)}{year_html}</div>'
            f'<div class="detail-genres">{genre_chips}</div>'
            f'<div class="detail-meta">{"".join(meta_chips)}</div>',
            unsafe_allow_html=True,
        )

        if pd.notna(row.get("avg_rating", np.nan)) and pd.notna(
            row.get("n_ratings", np.nan)
        ):
            st.caption(
                f"MovieLens users rated it "
                f'{float(row["avg_rating"]):.1f} on average across '
                f'{int(row["n_ratings"]):,} ratings.'
            )


def render_recommendations(
    model,
    catalog,
    all_ratings,
    user_to_idx,
    movie_to_idx,
    genre_lookup,
    global_mean,
    api_key,
    profile_mode,
    existing_user_id,
):
    """The ✨ For you tab — the patched recommendation flow."""
    top_n = st.slider(
        "Number of recommendations",
        6,
        24,
        12,
        step=2,
        key="n_recs",
    )

    st.markdown("", unsafe_allow_html=True)

    if profile_mode == EXISTING_USER_LABEL:
        user_id = int(existing_user_id or 0)

        if user_id not in user_to_idx:
            st.error(
                f"userId {user_id} is not in the training data — pick an id "
                f"between 1 and {max(user_to_idx):,} in the sidebar."
            )
            return

        top, affinity, n_candidates, history = recommend_for_existing_user(
            model,
            all_ratings,
            catalog,
            user_id,
            user_to_idx,
            genre_lookup,
            global_mean,
            top_n,
        )

        ucol1, ucol2, ucol3, ucol4 = st.columns(4)

        ucol1.metric("Profile", f"#{user_id}")
        ucol2.metric(
            "Ratings on file",
            f"{0 if history is None else len(history):,}",
        )
        ucol3.metric("Catalog scored", f"{n_candidates:,}")
        ucol4.metric("Reranked", f"top {top_n}")

        if top is None or top.empty:
            st.info(
                "This profile has already rated essentially the whole "
                "catalog — nothing left to recommend."
            )
            return

        render_taste_chips(affinity)

        sec_header(
            "For this profile",
            "Top picks from the NCF model + taste reranker",
            chip=f"top {len(top)}",
        )

        history_ratings = {}

        if history is not None and len(history):
            history_ratings = dict(
                zip(
                    history["movieId"].astype(int),
                    history["rating"].astype(float),
                )
            )

        render_movie_grid(
            top,
            api_key,
            ranked=True,
            user_ratings=history_ratings,
        )

        if history is not None and len(history):
            favs = history.sort_values("rating", ascending=False).head(6)

            favs = favs.merge(
                catalog[
                    ["movieId", "title", "genres", "avg_rating",
                     "n_ratings", "popularity"]
                ],
                on="movieId",
                how="left",
            ).dropna(subset=["title"])

            if not favs.empty:
                sec_header(
                    "Recently loved",
                    "This profile's highest-rated movies",
                )

                render_movie_grid(
                    favs,
                    api_key,
                    ranked=False,
                    user_ratings=history_ratings,
                    compact=True,
                )

        return

    # ---- new user flow ----
    user_ratings = dict(st.session_state.get("user_ratings", {}))

    if len(user_ratings) < MIN_NEW_USER_RATINGS:
        st.warning(
            f"You've rated {len(user_ratings)} of "
            f"{MIN_NEW_USER_RATINGS} movies needed for recommendations."
        )
        st.info(
            "Head to the ⭐ Rate movies tab — the quick-start grid gets "
            "you there in a minute."
        )
        return

    top, affinity, n_candidates, _ = recommend_for_new_user(
        model,
        catalog,
        movie_to_idx,
        genre_lookup,
        user_ratings,
        global_mean,
        top_n,
    )

    if top is None or top.empty:
        st.info(
            "No recommendations could be computed — try rating a few "
            "more movies."
        )
        return

    render_taste_chips(affinity)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    mcol1.metric("Your ratings", f"{len(user_ratings)}")
    mcol2.metric("Catalog scored", f"{n_candidates:,}")
    mcol3.metric(
        "Best match",
        f"{float(top.iloc[0]['match_score']) * 100:.0f}%",
    )
    mcol4.metric(
        "Avg predicted",
        f"{float(top['predicted_rating'].mean()):.2f} / 5",
    )

    sec_header(
        "For you",
        "Fresh from the NCF model + taste reranker",
        chip=f"top {len(top)}",
    )

    render_movie_grid(
        top,
        api_key,
        ranked=True,
        user_ratings=user_ratings,
    )


def main():
    model = load_model(str(MODEL_PATH))

    (
        ratings,
        catalog,
        movies,
        user_to_idx,
        movie_to_idx,
        genres,
        global_mean,
    ) = load_data(str(DATA_PATH))

    catalog = add_year_column(catalog)

    genre_lookup = build_genre_lookup(catalog)

    render_brand()

    api_key = st.sidebar.text_input(
        "TMDB API key (posters only)",
        value=TMDB_API_KEY,
        key="tmdb_api_key",
    )

    st.sidebar.markdown(
        '<div class="taste-label" style="margin:14px 0 8px;">'
        "Who&#8217;s watching</div>",
        unsafe_allow_html=True,
    )

    profile_mode = st.sidebar.radio(
        "Profile",
        [NEW_USER_LABEL, EXISTING_USER_LABEL],
        key="profile_mode",
        label_visibility="collapsed",
    )

    existing_user_id = None

    if profile_mode == EXISTING_USER_LABEL:
        existing_user_id = st.sidebar.number_input(
            "MovieLens userId",
            min_value=1,
            max_value=int(max(user_to_idx)),
            value=1,
            step=1,
            key="existing_uid",
        )

        st.sidebar.caption(
            "Any userId present in the training data — the model scores "
            "it with its learned embedding directly."
        )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "NCF + genre-affinity reranking · posters via TMDB · "
        "dark cinematic UI"
    )

    render_hero(
        len(catalog),
        len(user_to_idx),
        len(ratings),
    )

    tab_rate, tab_catalog, tab_foryou = st.tabs(
        ["⭐ Rate movies", "🎬 Browse catalog", "✨ For you"]
    )

    user_ratings = dict(st.session_state.get("user_ratings", {}))

    with tab_rate:
        render_quick_start(catalog, api_key, user_ratings)

        st.markdown("---")

        render_search_and_rate(catalog, api_key, user_ratings)

        st.markdown("---")

        render_your_ratings(catalog, user_ratings)

    with tab_catalog:
        render_catalog(catalog, genres, api_key, user_ratings)

    with tab_foryou:
        render_recommendations(
            model,
            catalog,
            ratings,
            user_to_idx,
            movie_to_idx,
            genre_lookup,
            global_mean,
            api_key,
            profile_mode,
            existing_user_id,
        )


main()
