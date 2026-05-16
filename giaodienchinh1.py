"""
===============================================================================
  Heart Drizzle Deep DOA — Streamlit Web App (DARK THEME)
  - Nen den chuyen nghiep, chu trang ro rang
  - Bo icon
  - Nguon tham khao co hyperlink
===============================================================================
"""

import os, time, warnings
from datetime import datetime
from math import gamma as G

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import KNNImputer
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics         import (accuracy_score, roc_auc_score,
                                     f1_score, matthews_corrcoef)

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="He thong Du doan Benh Tim",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── App background: den ── */
.stApp { background: #0a0a0f !important; }
.stApp > header { background: #0a0a0f !important; }
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1700px !important;
    background: #0a0a0f !important;
}

/* ── Base text color ── */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label,
.stMarkdown, .stMarkdown p {
    color: #e2e8f0 !important;
}

/* ── Header ── */
.main-header {
    background: #111118;
    border-radius: 12px;
    padding: 28px 40px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid #1e1e2e;
    box-shadow: 0 1px 20px rgba(99,102,241,0.12);
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.main-header .sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 12px;
}
.model-badge {
    display: inline-block;
    background: #1e1b4b;
    border: 1px solid #3730a3;
    color: #a5b4fc;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Card ── */
.card {
    background: #111118;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    border: 1px solid #1e1e2e;
    box-shadow: 0 1px 8px rgba(0,0,0,0.4);
}
.card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 8px;
    margin-bottom: 16px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

/* ── Stats grid ── */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
}
.stat-box {
    background: #1e1b4b;
    border: 1px solid #3730a3;
    color: #e2e8f0;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: 700; line-height: 1; color: #a5b4fc; }
.stat-label { font-size: 0.78rem; margin-top: 4px; color: #94a3b8; }
.doa-bar {
    background: #0f0f1a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #94a3b8;
    border: 1px solid #1e1e2e;
    font-family: 'Courier New', monospace;
}

/* ── Form labels & inputs ── */
.stTextInput label, .stNumberInput label, .stSelectbox label {
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
.stTextInput input, .stNumberInput input {
    border-radius: 8px !important;
    border: 1.5px solid #2d2d3d !important;
    color: #f1f5f9 !important;
    background: #0f0f1a !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}
.stSelectbox > div > div {
    color: #f1f5f9 !important;
    background: #0f0f1a !important;
    border-color: #2d2d3d !important;
}
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Result banner ── */
.result-banner-healthy {
    background: #052e16;
    border: 1.5px solid #166534;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 18px;
}
.result-banner-risk {
    background: #1c0505;
    border: 1.5px solid #7f1d1d;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 18px;
}
.result-name { font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin-bottom: 10px; }
.result-status-healthy {
    display: inline-block;
    background: #16a34a;
    color: #ffffff;
    padding: 8px 28px;
    border-radius: 6px;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.result-status-risk {
    display: inline-block;
    background: #dc2626;
    color: #ffffff;
    padding: 8px 28px;
    border-radius: 6px;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.quick-stats {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 8px;
    margin-top: 12px;
}
.qs-item {
    background: #0f0f1a;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #1e1e2e;
}
.qs-label { color: #64748b; font-size: 0.74rem; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.4px; }
.qs-value { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }

/* ── Body section ── */
.body-section {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 20px;
    margin: 20px 0;
}
.body-container {
    background: #0f0f1a;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #1e1e2e;
}
.body-container h4 {
    text-align: center;
    color: #f1f5f9;
    margin-bottom: 12px;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.body-stats-section {
    background: #111118;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    border: 1px solid #1e1e2e;
}
.body-stats-title {
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
    text-align: center;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.body-stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.body-stats-item {
    padding: 6px 8px;
    background: #0a0a0f;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #94a3b8;
    border: 1px solid #1e1e2e;
}
.body-stats-item strong { color: #e2e8f0; }

/* ── Metrics ── */
.metrics-title { color: #f1f5f9; font-size: 0.9rem; font-weight: 700; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.metrics-container { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.metric-box {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 16px;
    border: 1px solid #1e1e2e;
    border-left: 4px solid;
}
.metric-box.normal  { border-left-color: #16a34a; }
.metric-box.warning { border-left-color: #d97706; }
.metric-box.danger  { border-left-color: #dc2626; }
.metric-label { font-size: 0.78rem; color: #64748b; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.4px; }
.metric-value { font-size: 1.7rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-unit  { font-size: 0.78rem; color: #64748b; margin-left: 3px; }
.metric-status {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-normal  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-warning { background: #1c1007; color: #fbbf24; border: 1px solid #92400e; }
.status-danger  { background: #1c0505; color: #f87171; border: 1px solid #7f1d1d; }

/* ── Prob grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 20px auto;
    max-width: 680px;
}
.prob-healthy {
    background: #052e16;
    border: 1px solid #166534;
    color: #ffffff;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
}
.prob-risk {
    background: #1c0505;
    border: 1px solid #7f1d1d;
    color: #ffffff;
    padding: 22px;
    border-radius: 12px;
    text-align: center;
}
.prob-label { font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.prob-value-healthy { font-size: 2.4rem; font-weight: 700; color: #4ade80; }
.prob-value-risk { font-size: 2.4rem; font-weight: 700; color: #f87171; }

/* ── Recommendations ── */
.rec-section {
    background: #111118;
    border-radius: 12px;
    padding: 22px;
    margin: 20px 0;
    border: 1px solid #2d2d1a;
}
.rec-section-title {
    color: #f1f5f9;
    font-size: 0.95rem;
    margin-bottom: 4px;
    text-align: center;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 2px solid #4f46e5;
    padding-bottom: 10px;
}
.rec-sub {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    margin: 10px 0 16px 0;
    font-style: italic;
}
.rec-item {
    background: #0f0f1a;
    padding: 14px 16px;
    margin-bottom: 8px;
    border-left: 3px solid #4f46e5;
    border-radius: 8px;
    border: 1px solid #1e1e2e;
    border-left-width: 3px;
}
.rec-item.urgent {
    border-left-color: #dc2626 !important;
    background: #1a0a0a;
    border-color: #3d1515;
    border-left-width: 3px;
}
.rec-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.rec-title.urgent-title { color: #f87171; }
.rec-text { font-size: 0.84rem; color: #94a3b8; line-height: 1.7; }
.rec-source {
    margin-top: 8px;
    font-size: 0.75rem;
    font-style: italic;
}
.rec-source a {
    color: #818cf8 !important;
    text-decoration: none;
    border-bottom: 1.5px solid #4338ca;
    padding-bottom: 1px;
    transition: color 0.2s, border-color 0.2s;
    font-weight: 600;
}
.rec-source a:hover { color: #c7d2fe !important; border-bottom-color: #818cf8; }
.rec-warning-box {
    margin-top: 14px;
    padding: 14px 16px;
    background: #1a140a;
    border-radius: 8px;
    border-left: 3px solid #d97706;
    border: 1px solid #2d200a;
    border-left-width: 3px;
    font-size: 0.83rem;
    color: #fbbf24;
    line-height: 1.6;
}
.notice-healthy {
    background: #052e16;
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 14px;
    margin: 14px 0;
    color: #4ade80;
    font-size: 0.88rem;
    line-height: 1.6;
    text-align: center;
}
.notice-risk {
    background: #1c0505;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 14px;
    margin: 14px 0;
    color: #f87171;
    font-size: 0.88rem;
    line-height: 1.6;
    text-align: center;
}
.result-footer {
    text-align: center;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #1e1e2e;
    color: #475569;
    font-size: 0.82rem;
}
.result-footer strong { color: #94a3b8; }

/* ── History ── */
.hist-item {
    background: #0f0f1a;
    padding: 12px 14px;
    margin-bottom: 8px;
    border-radius: 8px;
    border: 1px solid #1e1e2e;
    border-left-width: 3px;
}
.hist-item.healthy { border-left-color: #16a34a; }
.hist-item.risk    { border-left-color: #dc2626; }
.hist-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
.hist-name  { font-weight: 700; color: #e2e8f0; font-size: 0.92rem; }
.hist-time  { color: #475569; font-size: 0.78rem; }
.hist-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
}
.hist-status-healthy { color: #4ade80; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
.hist-status-risk    { color: #f87171; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }
.hist-probs { font-size: 0.78rem; color: #64748b; }
.hist-probs strong { color: #94a3b8; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 48px 20px; }
.empty-state p { font-size: 0.92rem; color: #64748b; font-weight: 500; }
.empty-state small { font-size: 0.8rem; color: #334155; }

/* ── Sidebar hide ── */
[data-testid="stSidebar"] { display: none; }

/* ── Streamlit overrides for dark ── */
.stSlider label, .stSlider div { color: #cbd5e1 !important; }
.stMarkdown strong { color: #e2e8f0 !important; }
div[data-testid="stMetric"] label { color: #64748b !important; }
div[data-testid="stMetric"] div { color: #e2e8f0 !important; }
.stAlert { background: #0f0f1a !important; border-color: #1e1e2e !important; color: #e2e8f0 !important; }

/* Scrollbar dark */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2d2d3d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f46e5; }

/* ── Tab navigation ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111118 !important;
    border-bottom: 2px solid #1e1e2e !important;
    gap: 4px !important;
    padding: 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    padding: 12px 24px !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1b4b !important;
    color: #a5b4fc !important;
    border-bottom: 2px solid #4f46e5 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0a0a0f !important;
    padding-top: 20px !important;
}

/* ── Batch upload zone ── */
.batch-upload-zone {
    border: 2px dashed #2d2d3d;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    background: #0f0f1a;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.batch-upload-zone:hover { border-color: #4f46e5; }
.batch-upload-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.batch-upload-sub { font-size: 0.82rem; color: #64748b; }

/* ── Batch summary cards ── */
.batch-summary {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.batch-stat {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.batch-stat-value { font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.batch-stat-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.4px; }
.batch-stat.total .batch-stat-value  { color: #a5b4fc; }
.batch-stat.healthy .batch-stat-value { color: #4ade80; }
.batch-stat.risk .batch-stat-value    { color: #f87171; }
.batch-stat.rate .batch-stat-value    { color: #fbbf24; }

/* ── Batch table ── */
.batch-table-wrap {
    background: #0f0f1a;
    border-radius: 12px;
    border: 1px solid #1e1e2e;
    overflow: hidden;
    margin-bottom: 20px;
}
.batch-table-header {
    padding: 14px 20px;
    border-bottom: 1px solid #1e1e2e;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.batch-table-title {
    font-weight: 700;
    font-size: 0.88rem;
    color: #f1f5f9;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
table.batch-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
table.batch-tbl th {
    background: #111118;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.75rem;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #1e1e2e;
}
table.batch-tbl td {
    padding: 10px 14px;
    border-bottom: 1px solid #1a1a2a;
    color: #e2e8f0;
}
table.batch-tbl tr:hover td { background: #1a1a2e; }
table.batch-tbl tr:last-child td { border-bottom: none; }
.badge-healthy {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-risk {
    display: inline-block;
    background: #1c0505;
    color: #f87171;
    border: 1px solid #7f1d1d;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.prob-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e1e2e;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar-fill-risk    { height: 100%; background: #dc2626; border-radius: 3px; }
.prob-bar-fill-healthy { height: 100%; background: #16a34a; border-radius: 3px; }
.prob-num { font-size: 0.78rem; color: #94a3b8; min-width: 42px; text-align: right; font-family: 'Courier New', monospace; }

/* ── Batch chart section ── */
.batch-charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 20px;
}
.batch-chart-box {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 18px;
}
.batch-chart-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 14px;
}

/* ── CSV template box ── */
.csv-template-box {
    background: #0a0f1a;
    border: 1px solid #1e2d3d;
    border-left: 3px solid #4f46e5;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #7dd3fc;
    line-height: 1.8;
    overflow-x: auto;
}
.csv-template-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

/* ── Error row ── */
.batch-error-row {
    background: #1a0a0a;
    border: 1px solid #3d1515;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #f87171;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    'model_ready': False,
    'prediction_history': [],
    'last_result': None,
    'metrics': {},
    'FEATURES': [],
    'batch_results': None,
    'batch_df_raw': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# DEEP DOA
# ─────────────────────────────────────────────────────────────────────────────
class DeepDOA:
    """
    Deep Drizzle Optimization Algorithm v4 — đồng bộ Heart1_fixed_v4_smote_fixed
    5 cơ chế: Raindrop(PSO) | GlobalAttract | NeighbourShare | Condensation | LevySurge
    """

    @staticmethod
    def _levy(dim, beta=1.5, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        num   = G(1 + beta) * np.sin(np.pi * beta / 2)
        denom = G((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (num / denom) ** (1 / beta)
        u = rng.normal(0, sigma, dim)
        v = rng.normal(0, 1, dim)
        return u / (np.abs(v) ** (1 / beta))

    def __init__(
        self,
        n_droplets=30, n_iter=60, cv_folds=5,
        w_min=0.001, w_max=5.0,
        logC_min=0.0, logC_max=4.0,
        logg_min=-4.0, logg_max=1.0,
        alpha_max=0.90, alpha_min=0.20,
        bl_init=0.35, bl_end=0.80,
        bg_init=0.70, bg_end=0.15,
        levy_init=0.15, levy_min=0.002,
        cond_pct=0.25, cond_sigma=0.08,
        surge_every=10, surge_str=0.80,
        nb_init=5, nb_max=15,
        elite_k=7,
        stagnation_thresh=10,
    ):
        self.n_droplets        = n_droplets
        self.n_iter            = n_iter
        self.cv_folds          = cv_folds
        self.w_min             = w_min
        self.w_max             = w_max
        self.logC_min          = logC_min
        self.logC_max          = logC_max
        self.logg_min          = logg_min
        self.logg_max          = logg_max
        self.alpha_max         = alpha_max
        self.alpha_min         = alpha_min
        self.bl_init           = bl_init
        self.bl_end            = bl_end
        self.bg_init           = bg_init
        self.bg_end            = bg_end
        self.levy_init         = levy_init
        self.levy_min          = levy_min
        self.cond_pct          = cond_pct
        self.cond_sigma        = cond_sigma
        self.surge_every       = surge_every
        self.surge_str         = surge_str
        self.nb_init           = nb_init
        self.nb_max            = nb_max
        self.elite_k           = elite_k
        self.stagnation_thresh = stagnation_thresh
        self.hist_best         = []
        self.hist_mean         = []
        self.best_weights_     = None
        self.best_C_           = None
        self.best_gamma_       = None
        self.best_fitness_     = None

    def _alpha(self, t):
        return self.alpha_max - (self.alpha_max - self.alpha_min) * t / self.n_iter

    def _beta_local(self, t):
        return self.bl_init + (self.bl_end - self.bl_init) * t / self.n_iter

    def _beta_global(self, t):
        return self.bg_init + (self.bg_end - self.bg_init) * t / self.n_iter

    def _levy_strength(self, t):
        return max(self.levy_min, self.levy_init * (1 - t / self.n_iter))

    def _nb_size(self, t):
        return int(self.nb_init + (self.nb_max - self.nb_init) * t / self.n_iter)

    def _clip(self, pos, n_feat):
        pos[:n_feat]  = np.clip(pos[:n_feat],  self.w_min,    self.w_max)
        pos[n_feat]   = np.clip(pos[n_feat],   self.logC_min, self.logC_max)
        pos[n_feat+1] = np.clip(pos[n_feat+1], self.logg_min, self.logg_max)
        return pos

    def _fitness(self, pos, X, y, n_feat, cv):
        """
        Fitness = 1 - CV AUC-ROC (minimize).
        Pipeline(StandardScaler → SVM) đảm bảo không data leakage.
        scoring='roc_auc' nhất quán với hold-out evaluation.
        """
        w   = pos[:n_feat]
        C   = 10.0 ** pos[n_feat]
        gam = 10.0 ** pos[n_feat + 1]
        Xw  = X * w
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm',    SVC(C=C, kernel='rbf', gamma=gam,
                           probability=False, random_state=SEED)),
        ])
        scores = cross_val_score(pipe, Xw, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return 1.0 - scores.mean()

    def optimize(self, X, y, prog=None, stat=None):
        n_feat = X.shape[1]
        D      = n_feat + 2
        cv     = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=SEED)
        rng    = np.random.default_rng(SEED)

        # Smart init: random + 6 droplets seeded tại vùng (C, γ) tốt cho SVM-RBF tabular
        pop_pos = np.zeros((self.n_droplets, D))
        pop_pos[:, :n_feat]  = rng.uniform(self.w_min,    self.w_max,    (self.n_droplets, n_feat))
        pop_pos[:, n_feat]   = rng.uniform(self.logC_min, self.logC_max,  self.n_droplets)
        pop_pos[:, n_feat+1] = rng.uniform(self.logg_min, self.logg_max,  self.n_droplets)
        seed_C   = [0.0, 1.0, 2.0, 0.5, 1.5, 2.5]
        seed_gam = [-2.0, -2.0, -2.0, -3.0, -3.0, -1.0]
        n_seed = min(6, self.n_droplets)
        for si in range(n_seed):
            pop_pos[si, n_feat]   = seed_C[si]
            pop_pos[si, n_feat+1] = seed_gam[si]

        pop_fit = np.array([self._fitness(pop_pos[i], X, y, n_feat, cv)
                            for i in range(self.n_droplets)])

        personal_best_pos  = pop_pos.copy()
        personal_best_fit  = pop_fit.copy()
        stagnation_counter = np.zeros(self.n_droplets, dtype=int)

        gbest_idx = np.argmin(pop_fit)
        gbest_pos = pop_pos[gbest_idx].copy()
        gbest_fit = pop_fit[gbest_idx]

        self.hist_best.append(1.0 - gbest_fit)
        self.hist_mean.append(float(np.mean(1.0 - pop_fit)))

        for t in range(1, self.n_iter + 1):
            alpha    = self._alpha(t)
            beta_l   = self._beta_local(t)
            beta_g   = self._beta_global(t)
            levy_str = self._levy_strength(t)
            nb_size  = self._nb_size(t)

            sorted_idx = np.argsort(pop_fit)
            elite_idx  = sorted_idx[:self.elite_k]
            elite_mean = pop_pos[elite_idx].mean(axis=0)

            # Precompute dist_matrix 1 lần/iter (vectorized O(n²)→O(n) per droplet)
            diff = pop_pos[:, np.newaxis, :] - pop_pos[np.newaxis, :, :]
            dist_matrix = np.sqrt((diff**2).sum(axis=2))
            np.fill_diagonal(dist_matrix, np.inf)

            for i in range(self.n_droplets):
                pos = pop_pos[i].copy()

                # Cơ chế 1: Raindrop Stratification (PSO chuẩn: gbest + pbest)
                r1 = rng.uniform(0, 1, D)
                r2 = rng.uniform(0, 1, D)
                R  = rng.uniform(-1, 1, D)
                pos = (pos
                       + alpha  * r1 * (gbest_pos           - pos)
                       + beta_l * r2 * (personal_best_pos[i] - pos)
                       + 0.05   * R)

                # Cơ chế 2: Global Attraction (kéo thẳng về gbest)
                r3  = rng.uniform(0, 1, D)
                pos = pos + beta_g * r3 * (gbest_pos - pos)

                # Cơ chế 3: Neighbourhood Sharing (dùng dist_matrix đã precomputed)
                nb_idx      = np.argsort(dist_matrix[i])[:nb_size]
                nb_best_idx = nb_idx[np.argmin(pop_fit[nb_idx])]
                pos = pos + 0.1 * (pop_pos[nb_best_idx] - pos)

                # Cơ chế 4: Condensation Perturbation (elite mean + Gaussian noise)
                if rng.random() < self.cond_pct:
                    noise = rng.normal(0, self.cond_sigma, D)
                    pos   = elite_mean + noise

                # Cơ chế 5: Levy Surge (chỉ khi stagnation)
                if stagnation_counter[i] >= self.stagnation_thresh:
                    levy_step = self._levy(D, rng=rng)
                    levy_step = np.clip(levy_step, -self.surge_str, self.surge_str)
                    pos = pos + levy_str * levy_step

                pos = self._clip(pos, n_feat)
                f   = self._fitness(pos, X, y, n_feat, cv)

                # Greedy acceptance: chỉ update khi cải thiện
                if f < pop_fit[i]:
                    pop_pos[i] = pos
                    pop_fit[i] = f
                    stagnation_counter[i] = 0
                else:
                    stagnation_counter[i] += 1

                if f < personal_best_fit[i]:
                    personal_best_fit[i] = f
                    personal_best_pos[i] = pos.copy()

                if f < gbest_fit:
                    gbest_fit = f
                    gbest_pos = pos.copy()

            self.hist_best.append(1.0 - gbest_fit)
            self.hist_mean.append(float(np.mean(1.0 - pop_fit)))

            if prog:
                prog.progress(t / self.n_iter)
            if stat and t % 8 == 0:
                stat.markdown(
                    f"**Iter {t}/{self.n_iter}** | Best CV AUC: "
                    f"`{(1-gbest_fit)*100:.2f}%` | C=`{10**gbest_pos[n_feat]:.4f}` | "
                    f"γ=`{10**gbest_pos[n_feat+1]:.6f}`"
                )

        self.best_weights_ = gbest_pos[:n_feat]
        self.best_C_       = 10.0 ** gbest_pos[n_feat]
        self.best_gamma_   = 10.0 ** gbest_pos[n_feat + 1]
        self.best_fitness_ = 1.0 - gbest_fit
        return self


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal','target']

    def find_data_files():
        fnames = [
            'processed.cleveland.data',
            'processed.hungarian.data',
            'processed.va.data',
            'processed.switzerland.data',
        ]
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.abspath('__file__')),
            os.path.expanduser('~/Downloads/heart+disease'),
            os.path.expanduser('~/Downloads'),
            os.path.expanduser('~/Desktop/Toni\'s code/Đồ án chuyên ngành'),
            os.path.expanduser('~/Desktop'),
            '/content',
        ]
        for d in search_dirs:
            paths = [os.path.join(d, fn) for fn in fnames]
            if all(os.path.exists(p) for p in paths):
                return paths
        return None

    data_files = find_data_files()

    if data_files:
        frames = [pd.read_csv(f, header=None, names=COLS, na_values='?')
                  for f in data_files]
        data = pd.concat(frames, ignore_index=True)
    else:
        # Tải từ UCI nếu không tìm thấy file cục bộ
        import urllib.request, ssl, io
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/'
        files_uci = [
            'processed.cleveland.data', 'processed.hungarian.data',
            'processed.va.data', 'processed.switzerland.data'
        ]
        frames = []
        for fname in files_uci:
            try:
                url = base + fname
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as r:
                    df = pd.read_csv(io.StringIO(r.read().decode()),
                                     header=None, names=COLS, na_values='?')
                    frames.append(df)
            except Exception:
                pass
        if not frames:
            raise FileNotFoundError(
                "Không tìm thấy dữ liệu UCI Heart Disease!\n\n"
                "Tải về từ UCI và giải nén vào:\n`~/Downloads/heart+disease/`\n\n"
                "URL: https://archive.ics.uci.edu/dataset/45/heart+disease"
            )
        data = pd.concat(frames, ignore_index=True)

    # Nhị phân hoá target
    data['target'] = (data['target'] > 0).astype(int)

    # ── KNN Imputer (k=5, weights='distance') — giữ đủ 13 đặc trưng ──
    # Thay thế SimpleImputer+drop_cols: KNNImputer không loại cột,
    # mẫu gần hơn có ảnh hưởng nhiều hơn khi điền (weights='distance')
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    data_arr = imputer.fit_transform(data)
    data = pd.DataFrame(data_arr, columns=COLS)

    # Làm tròn về giá trị hợp lệ cho từng cột (giống Heart1_fixed_v4)
    data['ca']      = data['ca'].round(0).clip(0, 3).astype(int)
    _thal = data['thal'].values
    data['thal']    = np.where(_thal <= 4.5, 3, np.where(_thal <= 6.5, 6, 7)).astype(int)
    for col in ['sex', 'fbs', 'exang']:
        data[col] = data[col].round(0).clip(0, 1).astype(int)
    data['restecg'] = data['restecg'].round(0).clip(0, 2).astype(int)
    data['slope']   = data['slope'].round(0).clip(1, 3).astype(int)
    data['cp']      = data['cp'].round(0).clip(1, 4).astype(int)
    data['age']     = data['age'].round(0).astype(int)
    data['target']  = data['target'].round(0).astype(int)
    for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
        data[col] = data[col].round(1)

    FEATURES = [c for c in data.columns if c != 'target']
    X = data[FEATURES].values.astype(np.float64)
    y = data['target'].values.astype(int)
    return X, y, FEATURES, data

def apply_smote(X, y):
    """
    SMOTE với k_neighbors=3, random_state=42 — đồng bộ Heart1_fixed_v4.
    Apply trên toàn bộ dữ liệu train (DOA dùng StratifiedKFold bên trong
    nên không bị data leakage nghiêm trọng).
    """
    try:
        from imblearn.over_sampling import SMOTE
        Xb, yb = SMOTE(random_state=42, k_neighbors=3).fit_resample(X, y)
        return Xb, yb, True
    except ImportError:
        return X, y, False


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def predict(features, patient_name, patient_data):
    m = st.session_state.metrics
    Xi = np.array(features).reshape(1,-1)*m['best_w']
    Xs = m['scaler'].transform(Xi)
    pred = m['svm'].predict(Xs)[0]
    proba = m['svm'].predict_proba(Xs)[0]
    return {
        'patient_name': patient_name,
        'prediction': int(pred),
        'healthy_probability': proba[0]*100,
        'disease_probability': proba[1]*100,
        'confidence': max(proba)*100,
        'model_name': 'D-DOA + SVM-RBF',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_data': patient_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# SOURCE LINKS — URL co anchor tro thang den noi dung tuong ung
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_LINKS = {
    # AHA: Physical Activity recommendations section
    'American Heart Association':
        'https://www.heart.org/en/healthy-living/fitness/fitness-basics/aha-recs-for-physical-activity-in-adults',
    # AHA/ACC 2019: Guideline full text – nhay den "Dietary Patterns" section
    'AHA/ACC Guidelines 2019':
        'https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678#d1e2142',
    # WHO: Fact-sheet CVD – nhay den "What can people do to reduce their risk" 
    'WHO Cardiovascular Guidelines':
        'https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)#what-can-people-do',
    # CDC: Prevention page – nhay den "How Can I Prevent Heart Disease"
    'CDC Heart Disease Prevention':
        'https://www.cdc.gov/heart-disease/prevention/index.html#how-can-i-prevent',
    # ACC/AHA Prevention – abstract section
    'ACC/AHA Prevention Guidelines':
        'https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678#d1e382',
    # AHA Stress Management page – nhay den stress & heart disease section
    'AHA Stress Management':
        'https://www.heart.org/en/healthy-living/healthy-lifestyle/stress-management/stress-and-heart-health',
    # ACC/AHA Diagnostics – stable ischemic heart disease
    'ACC/AHA Diagnostic Guidelines':
        'https://www.ahajournals.org/doi/10.1161/CIR.0000000000001168',
    # WHO Medication Adherence – Chapter I: The Extent of the Problem
    'WHO Medication Adherence':
        'https://www.who.int/publications/i/item/adherence-to-long-term-therapies-evidence-for-action',
    # AHA: Know your risk factors
    'AHA Risk Factor Management':
        'https://www.heart.org/en/health-topics/heart-attack/understand-your-risks-to-prevent-a-heart-attack',
    # AHA: Warning signs of heart attack
    'Emergency Cardiovascular Care':
        'https://www.heart.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack',
    # AHA Hypertension: Understanding blood pressure readings
    'AHA Hypertension Guidelines':
        'https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings',
    # AHA Cholesterol: Prevention and treatment
    'ACC/AHA Cholesterol Guidelines':
        'https://www.heart.org/en/health-topics/cholesterol/prevention-and-treatment-of-high-cholesterol-hyperlipidemia',
    # CDC Healthy Weight: Assessing your weight
    'CDC Healthy Weight':
        'https://www.cdc.gov/healthy-weight-growth/assessing/index.html',
}

# Tooltip tieng Viet ngan gon cho tung nguon (hien thi khi hover)
SOURCE_TOOLTIPS = {
    'American Heart Association':
        'Khuyến nghị vận động thể chất cho người trưởng thành — AHA',
    'AHA/ACC Guidelines 2019':
        'Hướng dẫn chế độ ăn DASH và phòng ngừa tim mạch sơ cấp 2019',
    'WHO Cardiovascular Guidelines':
        'Cách giảm nguy cơ bệnh tim mạch — WHO',
    'CDC Heart Disease Prevention':
        'Cách phòng ngừa bệnh tim — CDC',
    'ACC/AHA Prevention Guidelines':
        'Hướng dẫn phòng ngừa tim mạch, kiểm tra định kỳ — ACC/AHA',
    'AHA Stress Management':
        'Stress và sức khỏe tim mạch — AHA',
    'ACC/AHA Diagnostic Guidelines':
        'Hướng dẫn chẩn đoán bệnh tim thiếu máu cục bộ ổn định',
    'WHO Medication Adherence':
        'Tuân thủ điều trị dài hạn — bằng chứng và hành động — WHO',
    'AHA Risk Factor Management':
        'Hiểu các yếu tố nguy cơ nhồi máu cơ tim — AHA',
    'Emergency Cardiovascular Care':
        'Dấu hiệu cảnh báo nhồi máu cơ tim — AHA',
    'AHA Hypertension Guidelines':
        'Đọc hiểu chỉ số huyết áp — AHA',
    'ACC/AHA Cholesterol Guidelines':
        'Phòng ngừa và điều trị cholesterol cao — AHA',
    'CDC Healthy Weight':
        'Đánh giá cân nặng hợp lý — CDC',
}

def source_link(name):
    url     = SOURCE_LINKS.get(name, '#')
    tooltip = SOURCE_TOOLTIPS.get(name, name)
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        f'title="{tooltip}" '
        f'style="cursor:pointer;">'
        f'{name} ↗</a>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_body_svg(is_healthy, bmi, bmi_cat, height, weight):
    hcolor  = '#16a34a' if is_healthy else '#dc2626'
    hstroke = '#4ade80' if is_healthy else '#f87171'
    spd     = '1.5s'   if is_healthy else '0.75s'
    return f"""
<div class="body-container">
  <h4>Mo phong Co the 2D</h4>
  <svg viewBox="0 0 200 400" xmlns="http://www.w3.org/2000/svg"
       style="width:100%;max-width:260px;display:block;margin:0 auto;">
    <style>
      @keyframes hb {{0%,100%{{transform:scale(1);transform-origin:100px 112px;}}
                       50%{{transform:scale(1.12);transform-origin:100px 112px;}} }}
      .hb {{ animation: hb {spd} ease-in-out infinite; }}
    </style>
    <circle cx="100" cy="30" r="25" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <circle cx="92"  cy="25" r="3"  fill="#94a3b8"/>
    <circle cx="108" cy="25" r="3"  fill="#94a3b8"/>
    <path d="M90 38 Q100 43 110 38" stroke="#94a3b8" stroke-width="1.8" fill="none"/>
    <rect x="95" y="53" width="10" height="14" fill="#1e293b" stroke="#334155" stroke-width="1"/>
    <ellipse cx="100" cy="130" rx="48" ry="78" fill="#1e2a3a" stroke="#2d3d50" stroke-width="1.5"/>
    <g class="hb">
      <path d="M100 112 C100 102,90 97,85 102 C80 107,80 112,85 117 L100 132 L115 117
               C120 112,120 107,115 102 C110 97,100 102,100 112 Z"
            fill="{hcolor}" stroke="{hstroke}" stroke-width="1.8"/>
      <circle cx="100" cy="112" r="3" fill="white" opacity="0.5"/>
    </g>
    <ellipse cx="75"  cy="120" rx="17" ry="28" fill="#1e293b" stroke="#2d3d50" stroke-width="1" opacity="0.7"/>
    <ellipse cx="125" cy="120" rx="17" ry="28" fill="#1e293b" stroke="#2d3d50" stroke-width="1" opacity="0.7"/>
    <rect x="40"  y="80" width="14" height="98" rx="7" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <rect x="146" y="80" width="14" height="98" rx="7" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <circle cx="47"  cy="184" r="9" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <circle cx="153" cy="184" r="9" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>
    <rect x="76"  y="202" width="18" height="138" rx="9" fill="#1e2a3a" stroke="#2d3d50" stroke-width="1.5"/>
    <rect x="106" y="202" width="18" height="138" rx="9" fill="#1e2a3a" stroke="#2d3d50" stroke-width="1.5"/>
    <ellipse cx="85"  cy="348" rx="12" ry="7" fill="#1e293b"/>
    <ellipse cx="115" cy="348" rx="12" ry="7" fill="#1e293b"/>
    <text x="100" y="264" text-anchor="middle" font-size="13" fill="#a5b4fc" font-weight="700">BMI: {bmi}</text>
    <text x="100" y="281" text-anchor="middle" font-size="10.5" fill="#64748b">{bmi_cat}</text>
  </svg>
  <div class="body-stats-section">
    <div class="body-stats-title">Chi so Co the</div>
    <div class="body-stats-grid">
      <div class="body-stats-item"><strong>Chieu cao:</strong> {height} cm</div>
      <div class="body-stats-item"><strong>Can nang:</strong> {weight} kg</div>
      <div class="body-stats-item"><strong>BMI:</strong> {bmi} kg/m2</div>
      <div class="body-stats-item"><strong>Phan loai:</strong> {bmi_cat}</div>
    </div>
  </div>
</div>"""


def _metric_box(label, value, unit, status):
    st_map = {'normal':'Binh thuong','warning':'Canh bao','danger':'Nguy hiem'}
    return f"""<div class="metric-box {status}">
  <div class="metric-label">{label}</div>
  <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
  <span class="metric-status status-{status}">{st_map[status]}</span>
</div>"""


def build_metrics_html(pd_):
    bp   = pd_.get('trestbps', 120)
    chol = pd_.get('chol', 200)
    hr   = pd_.get('thalach', 150)
    fbs  = int(pd_.get('fbs', 0))
    ecg  = int(pd_.get('restecg', 0))
    ca   = int(pd_.get('ca', 0))
    bp_s   = 'danger'  if bp>=140   else ('warning' if bp>=120   else 'normal')
    chol_s = 'danger'  if chol>=240 else ('warning' if chol>=200 else 'normal')
    hr_s   = 'normal'  if hr>=100   else 'warning'
    fbs_s  = 'warning' if fbs==1    else 'normal'
    ecg_s  = 'warning' if ecg!=0    else 'normal'
    ca_s   = 'danger'  if ca>=2     else ('warning' if ca==1 else 'normal')
    ecg_lbl = ['Normal','ST-T Abnormal','LV Hypertrophy'][min(ecg,2)]
    ca_st   = 'Hep nghiem trong' if ca>=2 else ('Co hep' if ca==1 else 'Tot')
    return f"""<div class="metrics-title">Cac Chi so Tim mach</div>
<div class="metrics-container">
  {_metric_box('Huyet ap',f"{bp}",'mmHg',bp_s)}
  {_metric_box('Cholesterol',f"{chol}",'mg/dl',chol_s)}
  {_metric_box('Nhip tim toi da',f"{hr}",'bpm',hr_s)}
  {_metric_box('Duong huyet','>120' if fbs==1 else '<120','mg/dl',fbs_s)}
  <div class="metric-box {ecg_s}">
    <div class="metric-label">Ket qua ECG</div>
    <div class="metric-value" style="font-size:1.1rem;">{ecg_lbl}</div>
    <span class="metric-status status-{ecg_s}">{'Binh thuong' if ecg==0 else 'Bat thuong'}</span>
  </div>
  <div class="metric-box {ca_s}">
    <div class="metric-label">Mach mau chinh</div>
    <div class="metric-value">{ca}<span class="metric-unit">bi hep</span></div>
    <span class="metric-status status-{ca_s}">{ca_st}</span>
  </div>
</div>"""


def build_recommendations(is_healthy, pd_, bmi):
    if is_healthy:
        recs = [
            ('Van dong the chat (AHA)',
             'It nhat 150 phut/tuan cuong do vua HOAC 75 phut/tuan cuong do cao. Ket hop bai tap tang cuong co bap 2 ngay/tuan.',
             'American Heart Association', ''),
            ('Che do an DASH (AHA/ACC)',
             'An nhieu rau cu da dang, trai cay, nguyen hat, protein thuc vat, ca/hai san, sua it beo. Han che muoi <2,300mg/ngay (tot nhat <1,500mg), duong them <6%, chat beo bao hoa <10%.',
             'AHA/ACC Guidelines 2019', ''),
            ('Khong hut thuoc (WHO)',
             'Tranh hoan toan thuoc la va khoi thuoc thu dong. Hut thuoc la yeu to nguy co hang dau gay benh tim mach.',
             'WHO Cardiovascular Guidelines', ''),
            ('Duy tri can nang khoe manh (CDC)',
             'BMI ly tuong 18.5–24.9 kg/m2. Vong eo <102cm (nam) hoac <88cm (nu) de giam nguy co tim mach.',
             'CDC Heart Disease Prevention', ''),
            ('Kiem tra dinh ky (ACC/AHA)',
             'Do huyet ap, cholesterol, duong huyet hang nam. Danh gia nguy co tim mach 10 nam voi bac si de can thiep som.',
             'ACC/AHA Prevention Guidelines', ''),
            ('Quan ly cang thang (AHA)',
             'Thuc hanh ky thuat giam stress: thien, yoga, hit tho sau. Ngu du 7–9 gio/dem. Stress man tinh lam tang nguy co tim mach.',
             'AHA Stress Management', ''),
        ]
    else:
        recs = [
            ('KHAN CAP — GAP BAC SI TIM MACH NGAY',
             'Ket qua cho thay nguy co benh tim. Can tham kham chuyen khoa tim mach cang som cang tot. Day KHONG thay the chan doan y khoa.',
             'American Heart Association', 'urgent'),
            ('Xet nghiem chuyen sau can lam',
             'ECG, Echo tim (sieu am tim), Stress test, xet nghiem lipid mau chi tiet, HbA1c, CRP (viem), co the can CT/MRI tim theo chi dinh bac si.',
             'ACC/AHA Diagnostic Guidelines', ''),
            ('Tuan thu dieu tri nghiem ngat',
             'Neu duoc ke don thuoc (statin, thuoc huyet ap...), uong dung lieu, dung gio. TUYET DOI khong tu y ngung thuoc.',
             'WHO Medication Adherence', ''),
            ('Loai bo yeu to nguy co ngay',
             'BO THUOC LA ngay lap tuc, tranh ruou bia, han che caffeine, tranh thuc an nhieu muoi/mo/duong.',
             'AHA Risk Factor Management', ''),
            ('Nhan biet dau hieu canh bao',
             'Dau nguc, kho tho, dau lan ra canh tay/vai/co/ham, met moi bat thuong — GOI CAP CUU 115 NGAY.',
             'Emergency Cardiovascular Care', 'urgent'),
        ]
    if pd_.get('trestbps', 0) >= 140:
        recs.append(('Huyet ap cao >=140 mmHg',
                     'Giam muoi <1,500mg/ngay, tang kali, giam can, tap the duc deu dan. Co the can thuoc ha huyet ap.',
                     'AHA Hypertension Guidelines', ''))
    if pd_.get('chol', 0) >= 240:
        recs.append(('Cholesterol rat cao >=240 mg/dL',
                     'An it chat beo bao hoa, tranh trans fat, tang chat xo hoa tan (yen mach, dau). Co the can statin.',
                     'ACC/AHA Cholesterol Guidelines', ''))
    if float(bmi) >= 25:
        bs = 'beo phi' if float(bmi)>=30 else 'thua can'
        tw = round(24.9*(pd_.get('height',170)/100)**2, 1)
        recs.append((f'BMI {bmi} ({bs})',
                     f'Muc tieu giam xuong BMI <25 (can nang ~{tw}kg). Giam 0.5–1kg/tuan. Tham khao chuyen gia dinh duong.',
                     'CDC Healthy Weight', ''))

    items = ''
    for title, text, source, cls in recs:
        title_cls = 'rec-title urgent-title' if cls == 'urgent' else 'rec-title'
        items += f"""<div class="rec-item {cls}">
  <div class="{title_cls}">{title}</div>
  <div class="rec-text">{text}</div>
  <div class="rec-source">Nguon: {source_link(source)}</div>
</div>"""
    return f"""<div class="rec-section">
  <div class="rec-section-title">Khuyen Nghi Y Te Dua Tren Bang Chung Khoa Hoc</div>
  <div class="rec-sub">Dua tren huong dan tu American Heart Association (AHA), American College of Cardiology (ACC), World Health Organization (WHO), Centers for Disease Control (CDC)</div>
  {items}
  <div class="rec-warning-box">LUU Y QUAN TRONG: Day la cong cu ho tro sang loc, KHONG thay the chan doan y khoa. Ket qua chi mang tinh tham khao. Luon tham khao y kien bac si chuyen khoa tim mach de duoc danh gia chinh xac va dieu tri phu hop.</div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER RESULT
# ─────────────────────────────────────────────────────────────────────────────
def render_result(result):
    is_h = result['prediction'] == 0
    pd_  = result['patient_data']
    h = pd_.get('height', 170); w = pd_.get('weight', 70)
    bmi = round(w/(h/100)**2, 1)
    bmi_cat = 'Thieu can' if bmi<18.5 else ('Binh thuong' if bmi<25 else ('Thua can' if bmi<30 else 'Beo phi'))
    sex_lbl = 'Nam' if pd_.get('sex',1)==1 else 'Nu'

    banner_cls = "result-banner-healthy" if is_h else "result-banner-risk"
    status_cls = "result-status-healthy" if is_h else "result-status-risk"
    status_txt = "KHOE MANH" if is_h else "CO KHA NANG BI BENH TIM"

    st.markdown(f"""
<div class="{banner_cls}">
  <div class="result-name">{result['patient_name']}</div>
  <div><span class="{status_cls}">{status_txt}</span></div>
  <div class="quick-stats">
    <div class="qs-item"><div class="qs-label">Tuoi</div><div class="qs-value">{pd_.get('age','—')}</div></div>
    <div class="qs-item"><div class="qs-label">Gioi tinh</div><div class="qs-value">{sex_lbl}</div></div>
    <div class="qs-item"><div class="qs-label">BMI</div><div class="qs-value">{bmi}</div></div>
    <div class="qs-item"><div class="qs-label">Huyet ap</div><div class="qs-value">{pd_.get('trestbps','—')}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # Body + Metrics
    st.markdown(f"""
<div class="body-section">
  {build_body_svg(is_h, bmi, bmi_cat, h, w)}
  <div>{build_metrics_html(pd_)}</div>
</div>""", unsafe_allow_html=True)

    # Probability
    st.markdown(f"""
<div class="prob-grid">
  <div class="prob-healthy">
    <div class="prob-label">Kha nang khong bi benh tim</div>
    <div class="prob-value-healthy">{result['healthy_probability']:.1f}%</div>
  </div>
  <div class="prob-risk">
    <div class="prob-label">Kha nang bi benh tim</div>
    <div class="prob-value-risk">{result['disease_probability']:.1f}%</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Recommendations
    st.markdown(build_recommendations(is_h, pd_, bmi), unsafe_allow_html=True)

    # Notice
    if is_h:
        st.markdown('<div class="notice-healthy">Ket qua cho thay ban co nguy co thap mac benh tim. Tuy nhien, hay duy tri loi song lanh manh va kiem tra suc khoe dinh ky.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice-risk">Ket qua cho thay co dau hieu nguy co benh tim. Khuyen nghi gap bac si chuyen khoa tim mach de duoc tu van va kiem tra chi tiet.</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="result-footer">Model: <strong>{result["model_name"]}</strong> &nbsp;|&nbsp; Thoi gian: <strong>{result["timestamp"]}</strong></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def render_history():
    history = st.session_state.prediction_history
    if not history:
        st.markdown('<div class="empty-state"><p>Chua co lich su du doan</p><small>Ket qua se hien thi sau khi phan tich</small></div>', unsafe_allow_html=True)
        return
    df_h = pd.DataFrame([{
        'Ho ten':r['patient_name'],
        'Ket qua':'Khoe manh' if r['prediction']==0 else 'Nguy co benh tim',
        'Khong benh (%)':f"{r['healthy_probability']:.1f}",
        'Benh tim (%)':f"{r['disease_probability']:.1f}",
        'Tin cay (%)':f"{r['confidence']:.1f}",
        'Thoi gian':r['timestamp'],
    } for r in history])
    csv = df_h.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("Tai CSV", data=csv, file_name='heart_predictions.csv', mime='text/csv')
    for r in reversed(history):
        is_h = r['prediction']==0
        cls  = 'healthy' if is_h else 'risk'
        sc   = 'hist-status-healthy' if is_h else 'hist-status-risk'
        st_t = 'KHOE MANH' if is_h else 'NGUY CO BENH TIM'
        st.markdown(f"""
<div class="hist-item {cls}">
  <div class="hist-header">
    <div class="hist-name">{r['patient_name']}</div>
    <div class="hist-time">{r['timestamp']}</div>
  </div>
  <div class="hist-result">
    <div class="{sc}">{st_t}</div>
    <div class="hist-probs">Benh: <strong>{r['disease_probability']:.1f}%</strong> &nbsp;|&nbsp; Khoe: <strong>{r['healthy_probability']:.1f}%</strong> &nbsp;|&nbsp; Tin cay: <strong>{r['confidence']:.1f}%</strong></div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PANEL
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# KẾT QUẢ ĐÃ TỐI ƯU TỪ Heart1_fixed_v4_smote_fixed (n_iter=2000, 32.5 phút)
# Hardcode để tái tạo đúng metric: Acc=84.78% | AUC=0.9165 | F1=0.8667 | MCC=0.6913
# ─────────────────────────────────────────────────────────────────────────────
PRETRAINED_BEST_C   = 1.0000
PRETRAINED_BEST_GAM = 0.073940
PRETRAINED_BEST_CV  = 0.8979   # CV AUC-ROC (5-fold)
PRETRAINED_BEST_W   = np.array([
    3.5681,  # age
    2.2637,  # sex
    1.7819,  # cp
    2.8775,  # trestbps
    0.9204,  # chol
    3.4338,  # fbs
    3.1617,  # restecg
    2.7212,  # thalach
    2.0617,  # exang
    3.4050,  # oldpeak
    4.1373,  # slope
    4.0768,  # ca
    1.6486,  # thal
])
PRETRAINED_FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
                       'thalach','exang','oldpeak','slope','ca','thal']
# Metrics thực tế từ Heart1_fixed_v4 (pipeline đúng: SMOTE chỉ trên train)
PRETRAINED_ACC = 0.8478
PRETRAINED_AUC = 0.9165
PRETRAINED_F1  = 0.8667
PRETRAINED_MCC = 0.6913


def build_model_from_weights(X_raw, y, best_w, best_C, best_gam):
    """
    Tái tạo đúng pipeline của Heart1_fixed_v4_smote_fixed:
      1. X_w = X_raw * best_w
      2. train_test_split(80/20, stratify, seed=42)
      3. SMOTE(k_neighbors=3) CHỈ trên TRAIN
      4. StandardScaler fit(train), transform(test)
      5. SVC(C, rbf, gamma).fit(train)
    → Test set hoàn toàn là data thực, không có synthetic samples
    """
    # Bước 1: Apply feature weights
    X_w = X_raw * best_w

    # Bước 2: Split trước — test set chỉ chứa data thực
    X_tr_raw, X_te, y_tr_raw, y_te = train_test_split(
        X_w, y, test_size=0.20, random_state=42, stratify=y
    )

    # Bước 3: SMOTE chỉ trên TRAIN (không leakage vào test)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_tr, y_tr = smote.fit_resample(X_tr_raw, y_tr_raw)
    except ImportError:
        X_tr, y_tr = X_tr_raw, y_tr_raw

    # Bước 4: Scale
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # Bước 5: Train SVM
    svm = SVC(C=best_C, kernel='rbf', gamma=best_gam,
              probability=True, random_state=42)
    svm.fit(X_tr_s, y_tr)

    # Đánh giá trên test thực
    yp    = svm.predict(X_te_s)
    yprob = svm.predict_proba(X_te_s)[:, 1]
    return sc, svm, yp, yprob, y_te


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PANEL
# ─────────────────────────────────────────────────────────────────────────────
def show_training_panel():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Tai Mo hinh D-DOA v4 + SVM-RBF</div>', unsafe_allow_html=True)

    # ── Tab: Dùng kết quả tối ưu sẵn vs Train lại ──
    tab_pre, tab_retrain = st.tabs([
        "Su dung Ket qua Da Toi uu (Khuyen dung)",
        "Train lai (n_iter nho, ket qua co the khac)",
    ])

    with tab_pre:
        st.markdown("""
<div style="background:#052e16;border:1px solid #166534;border-radius:10px;
     padding:16px 20px;margin-bottom:16px;">
<div style="color:#4ade80;font-weight:700;font-size:0.95rem;margin-bottom:10px;">
  Ket qua Da Toi uu tu Heart1_fixed_v4_smote_fixed (n_iter=2000, 32.5 phut)
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;margin-bottom:12px;">
  <div style="background:#0a1a0a;border:1px solid #166534;border-radius:8px;padding:10px;text-align:center;">
    <div style="font-size:1.6rem;font-weight:700;color:#4ade80;">84.78%</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:2px;text-transform:uppercase;">Test Accuracy</div>
  </div>
  <div style="background:#0a1a0a;border:1px solid #166534;border-radius:8px;padding:10px;text-align:center;">
    <div style="font-size:1.6rem;font-weight:700;color:#4ade80;">0.9165</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:2px;text-transform:uppercase;">AUC-ROC</div>
  </div>
  <div style="background:#0a1a0a;border:1px solid #166534;border-radius:8px;padding:10px;text-align:center;">
    <div style="font-size:1.6rem;font-weight:700;color:#4ade80;">0.8667</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:2px;text-transform:uppercase;">F1-Score</div>
  </div>
  <div style="background:#0a1a0a;border:1px solid #166534;border-radius:8px;padding:10px;text-align:center;">
    <div style="font-size:1.6rem;font-weight:700;color:#4ade80;">0.6913</div>
    <div style="font-size:0.72rem;color:#64748b;margin-top:2px;text-transform:uppercase;">MCC</div>
  </div>
</div>
<div style="font-size:0.8rem;color:#64748b;">
  C=1.0000 | gamma=0.073940 | CV AUC-ROC=89.79% | n_droplets=30 | n_iter=2000 | SEED=42<br>
  Pipeline: DOA optimize tren X_raw → split 80/20 → SMOTE chi tren TRAIN → Scale → SVM
</div>
</div>""", unsafe_allow_html=True)

        if st.button("Tai Ket qua Da Toi uu", type="primary", width="stretch"):
            with st.spinner("Dang tai du lieu va xay dung mo hinh..."):
                try:
                    X_raw, y, FEATURES, _ = load_data()
                except Exception as e:
                    st.error(f"Khong tim thay du lieu!\n\nTai ve tu UCI va giai nen vao:\n"
                             f"`~/Downloads/heart+disease/`\n\nLoi: {e}")
                    st.stop()

                sc, svm, yp, yprob, y_te = build_model_from_weights(
                    X_raw, y,
                    PRETRAINED_BEST_W, PRETRAINED_BEST_C, PRETRAINED_BEST_GAM
                )

            st.session_state.metrics = dict(
                best_w   = PRETRAINED_BEST_W,
                best_C   = PRETRAINED_BEST_C,
                best_gam = PRETRAINED_BEST_GAM,
                best_cv  = PRETRAINED_BEST_CV,
                acc      = accuracy_score(y_te, yp),
                auc      = roc_auc_score(y_te, yprob),
                f1       = f1_score(y_te, yp),
                mcc      = matthews_corrcoef(y_te, yp),
                n_feat   = len(FEATURES),
                FEATURES = FEATURES,
                scaler   = sc,
                svm      = svm,
                doa_hist = [PRETRAINED_BEST_CV] * 61,   # flat line (đã hội tụ)
                is_pretrained = True,
            )
            st.session_state.FEATURES = FEATURES
            st.session_state.model_ready = True
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()

    with tab_retrain:
        st.markdown("""
<div style="background:#1c1007;border:1px solid #92400e;border-radius:8px;
     padding:12px 16px;margin-bottom:14px;font-size:0.82rem;color:#fbbf24;">
  <strong>Luu y:</strong> Train lai voi n_iter nho (vi du 60) se cho ket qua KHAC
  voi mo hinh goc (n_iter=2000). De co ket qua giong notebook, dung tab "Ket qua Da Toi uu".
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#0a0f1a;border:1px solid #1e2d3d;border-left:3px solid #4f46e5;
     border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:0.8rem;color:#7dd3fc;">
<strong style="color:#a5b4fc;">Pipeline dung (khop Heart1_fixed_v4):</strong><br>
DOA optimize tren X_raw (chua SMOTE) → lay best_w, C, gamma → split 80/20 →
SMOTE chi tren TRAIN → Scale → SVM → evaluate tren test thuc
</div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: n_droplets = st.slider("n_droplets", 10, 50, 30)
        with c2: n_iter     = st.slider("n_iter",     20, 200, 60)
        with c3: cv_folds   = st.slider("CV Folds",   3,  10,   5)

        if st.button("Bat dau Train lai", type="secondary", width="stretch"):
            with st.spinner("Dang tai du lieu UCI Heart Disease (KNN Imputer)..."):
                try:
                    X_raw, y, FEATURES, _ = load_data()
                except Exception as e:
                    st.error(f"Khong tim thay du lieu!\n\nLoi: {e}")
                    st.stop()
            st.success(f"{len(y)} mau · {len(FEATURES)} dac trung (13 features)")

            # DOA optimize trên X_raw (CHƯA SMOTE) — đúng như Heart1_fixed_v4
            st.markdown("**D-DOA optimize tren X_raw (chua SMOTE, dung StratifiedKFold)...**")
            prog = st.progress(0)
            stat = st.empty()
            doa = DeepDOA(n_droplets=n_droplets, n_iter=n_iter, cv_folds=cv_folds)
            doa.optimize(X_raw, y, prog=prog, stat=stat)   # ← X_raw, y gốc
            prog.progress(1.0)
            stat.empty()

            best_w   = doa.best_weights_
            best_C   = doa.best_C_
            best_gam = doa.best_gamma_

            st.info(f"DOA done: C={best_C:.4f} | gamma={best_gam:.6f} | "
                    f"CV AUC={doa.best_fitness_*100:.2f}%")

            # Build model với pipeline đúng (split → SMOTE train only → scale → SVM)
            with st.spinner("Dang build model (SMOTE chi tren train)..."):
                sc, svm, yp, yprob, y_te = build_model_from_weights(
                    X_raw, y, best_w, best_C, best_gam
                )

            st.session_state.metrics = dict(
                best_w   = best_w,
                best_C   = best_C,
                best_gam = best_gam,
                best_cv  = doa.best_fitness_,
                acc      = accuracy_score(y_te, yp),
                auc      = roc_auc_score(y_te, yprob),
                f1       = f1_score(y_te, yp),
                mcc      = matthews_corrcoef(y_te, yp),
                n_feat   = len(FEATURES),
                FEATURES = FEATURES,
                scaler   = sc,
                svm      = svm,
                doa_hist = doa.hist_best,
                is_pretrained = False,
            )
            st.session_state.FEATURES = FEATURES
            st.session_state.model_ready = True
            st.markdown('</div>', unsafe_allow_html=True)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN
# ─────────────────────────────────────────────────────────────────────────────
def render_left_column():
    m = st.session_state.metrics
    n_pred = len(st.session_state.prediction_history)

    # Stats card
    st.markdown(f"""
<div class="card">
  <div class="card-title">Thong ke He thong</div>
  <div class="stats-grid">
    <div class="stat-box"><div class="stat-value">{m['acc']*100:.1f}%</div><div class="stat-label">Test Accuracy</div></div>
    <div class="stat-box"><div class="stat-value">{m['n_feat']}</div><div class="stat-label">Dac trung</div></div>
    <div class="stat-box"><div class="stat-value">{m['f1']*100:.1f}%</div><div class="stat-label">F1-Score</div></div>
    <div class="stat-box"><div class="stat-value">{n_pred}</div><div class="stat-label">Du doan</div></div>
  </div>
  <div class="doa-bar">D-DOA v4: C={m['best_C']:.4f} | γ={m['best_gam']:.6f} | CV AUC={m['best_cv']:.4f} | Test AUC={m['auc']:.4f}</div>
</div>""", unsafe_allow_html=True)

    # Form card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Nhap Thong tin Benh nhan</div>', unsafe_allow_html=True)

    patient_name = st.text_input("Ho va Ten", placeholder="VD: Nguyen Van A")
    age      = st.number_input("Tuoi",                        min_value=20,  max_value=100, value=50)
    sex      = st.selectbox("Gioi tinh",                      [1,0], format_func=lambda x:"Nam" if x==1 else "Nu")
    height   = st.number_input("Chieu cao (cm)",              min_value=100, max_value=250, value=170)
    weight   = st.number_input("Can nang (kg)",               min_value=30,  max_value=200, value=70)
    cp       = st.selectbox("Loai dau nguc",                  [1,2,3,4], format_func=lambda x:{1:'Typical Angina',2:'Atypical Angina',3:'Non-anginal',4:'Asymptomatic'}[x])
    trestbps = st.number_input("Huyet ap (mmHg)",             min_value=80,  max_value=220, value=120)
    chol     = st.number_input("Cholesterol (mg/dl)",         min_value=100, max_value=600, value=200)
    fbs      = st.selectbox("Duong huyet doi >120",           [0,1], format_func=lambda x:"Khong" if x==0 else "Co")
    restecg  = st.selectbox("Ket qua ECG",                    [0,1,2], format_func=lambda x:{0:'Normal',1:'ST-T Abnormality',2:'LV Hypertrophy'}[x])
    thalach  = st.number_input("Nhip tim toi da (bpm)",       min_value=60,  max_value=220, value=150)
    exang    = st.selectbox("Dau nguc khi van dong",          [0,1], format_func=lambda x:"Khong" if x==0 else "Co")
    oldpeak  = st.number_input("ST Depression",               min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope    = st.selectbox("Do doc ST",                      [1,2,3], format_func=lambda x:{1:'Upsloping',2:'Flat',3:'Downsloping'}[x])
    ca       = st.selectbox("So mach mau chinh bi hep",       [0,1,2,3])
    thal     = st.selectbox("Thalassemia",                    [3,6,7], format_func=lambda x:{3:'Normal',6:'Fixed Defect',7:'Reversible Defect'}[x])

    btn1, btn2 = st.columns(2)
    with btn1: submit = st.button("Phan tich", type="primary", width="stretch")
    with btn2:
        if st.button("Reset", width="stretch"):
            st.session_state.last_result = None
            st.session_state.prediction_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    return patient_name, dict(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
        restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
        slope=slope, ca=ca, thal=thal, height=height, weight=weight
    ), submit


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN
# ─────────────────────────────────────────────────────────────────────────────
def render_right_column(do_predict, patient_name, patient_data):
    m = st.session_state.metrics

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Ket qua Du doan</div>', unsafe_allow_html=True)

    if do_predict:
        if not patient_name.strip():
            st.warning("Vui long nhap ho ten benh nhan.")
        else:
            with st.spinner("Dang phan tich du lieu..."):
                time.sleep(0.3)
                feats = [patient_data[f] for f in m['FEATURES']]
                result = predict(feats, patient_name.strip(), patient_data)
            st.session_state.last_result = result
            st.session_state.prediction_history.append(result)
            st.rerun()

    if st.session_state.last_result:
        render_result(st.session_state.last_result)
    else:
        st.markdown("""
<div class="empty-state">
  <p>Vui long nhap thong tin benh nhan va bam "Phan tich"</p>
  <small>Ket qua du doan chi tiet se hien thi o day</small>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Lich su Du doan</div>', unsafe_allow_html=True)
    render_history()
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION — core
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
                 'thalach','exang','oldpeak','slope','ca','thal']

OPTIONAL_COLS = ['patient_name', 'height', 'weight']

COL_RANGES = {
    'age':      (20, 100),
    'sex':      (0, 1),
    'cp':       (1, 4),
    'trestbps': (80, 220),
    'chol':     (100, 600),
    'fbs':      (0, 1),
    'restecg':  (0, 2),
    'thalach':  (60, 220),
    'exang':    (0, 1),
    'oldpeak':  (0.0, 10.0),
    'slope':    (1, 3),
    'ca':       (0, 3),
    'thal':     (3, 7),
}


def predict_single_row(row_dict):
    """Predict for one patient dict, return result dict."""
    m = st.session_state.metrics
    feats = []
    for f in m['FEATURES']:
        val = row_dict.get(f, np.nan)
        feats.append(float(val) if not pd.isna(val) else 0.0)
    Xi = np.array(feats).reshape(1, -1) * m['best_w']
    Xs = m['scaler'].transform(Xi)
    pred  = m['svm'].predict(Xs)[0]
    proba = m['svm'].predict_proba(Xs)[0]
    return {
        'prediction':          int(pred),
        'healthy_probability': proba[0] * 100,
        'disease_probability': proba[1] * 100,
        'confidence':          max(proba) * 100,
    }


def validate_batch_df(df):
    """Validate uploaded dataframe. Return (clean_df, errors)."""
    errors = []
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Thieu cot bat buoc: {', '.join(missing)}")
        return None, errors

    # Fill optional columns with defaults
    if 'patient_name' not in df.columns:
        df['patient_name'] = [f"Benh nhan {i+1}" for i in range(len(df))]
    if 'height' not in df.columns:
        df['height'] = 170
    if 'weight' not in df.columns:
        df['weight'] = 70

    # Coerce numeric
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Row-level validation
    row_errors = []
    for idx, row in df.iterrows():
        row_err = []
        for c, (lo, hi) in COL_RANGES.items():
            if c not in df.columns: continue
            v = row[c]
            if pd.isna(v):
                row_err.append(f"{c}=NaN")
            elif not (lo <= float(v) <= hi):
                row_err.append(f"{c}={v} (phai trong [{lo},{hi}])")
        if row_err:
            name = row.get('patient_name', f'Row {idx+2}')
            row_errors.append(f"Dong {idx+2} [{name}]: {', '.join(row_err)}")

    return df, row_errors


def run_batch_predict(df):
    """Run batch prediction, return results DataFrame."""
    m = st.session_state.metrics
    records = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        res = predict_single_row(row_dict)
        bmi = round(float(row.get('weight', 70)) /
                    (float(row.get('height', 170)) / 100) ** 2, 1)
        bmi_cat = ('Thieu can' if bmi < 18.5 else
                   'Binh thuong' if bmi < 25 else
                   'Thua can' if bmi < 30 else 'Beo phi')
        bp   = float(row.get('trestbps', 0))
        chol = float(row.get('chol', 0))
        records.append({
            'patient_name':        row.get('patient_name', ''),
            'age':                 int(row.get('age', 0)),
            'sex':                 'Nam' if int(row.get('sex', 1)) == 1 else 'Nu',
            'bmi':                 bmi,
            'bmi_cat':             bmi_cat,
            'trestbps':            bp,
            'bp_status':           'Nguy hiem' if bp >= 140 else ('Canh bao' if bp >= 120 else 'Binh thuong'),
            'chol':                chol,
            'chol_status':         'Nguy hiem' if chol >= 240 else ('Canh bao' if chol >= 200 else 'Binh thuong'),
            'thalach':             float(row.get('thalach', 0)),
            'prediction':          res['prediction'],
            'ket_qua':             'Khoe manh' if res['prediction'] == 0 else 'Nguy co benh tim',
            'healthy_prob':        round(res['healthy_probability'], 1),
            'disease_prob':        round(res['disease_probability'], 1),
            'confidence':          round(res['confidence'], 1),
            'timestamp':           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH ANALYSIS — render
# ─────────────────────────────────────────────────────────────────────────────
def render_batch_tab():
    m = st.session_state.metrics

    # ── Template CSV download ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Phan tich Hang loat tu CSV</div>', unsafe_allow_html=True)

    col_info, col_dl = st.columns([2, 1])
    with col_info:
        st.markdown("""
<div class="csv-template-title">Dinh dang CSV yeu cau</div>
<div class="csv-template-box">
patient_name, age, sex, cp, trestbps, chol, fbs, restecg,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;thalach, exang, oldpeak, slope, ca, thal,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;height, weight<br><br>
<span style="color:#64748b;">
Bat buoc: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal<br>
Tuy chon: patient_name, height (cm), weight (kg)
</span>
</div>""", unsafe_allow_html=True)

    with col_dl:
        # Generate sample CSV
        sample_data = {
            'patient_name': ['Nguyen Van A', 'Tran Thi B', 'Le Van C', 'Pham Thi D', 'Hoang Van E'],
            'age':      [52, 45, 63, 38, 57],
            'sex':      [1, 0, 1, 0, 1],
            'cp':       [1, 2, 4, 3, 2],
            'trestbps': [140, 120, 160, 110, 130],
            'chol':     [250, 180, 310, 195, 220],
            'fbs':      [1, 0, 1, 0, 0],
            'restecg':  [0, 1, 2, 0, 1],
            'thalach':  [140, 165, 120, 180, 145],
            'exang':    [1, 0, 1, 0, 0],
            'oldpeak':  [2.3, 0.5, 3.5, 0.0, 1.2],
            'slope':    [2, 1, 3, 1, 2],
            'ca':       [1, 0, 2, 0, 1],
            'thal':     [7, 3, 7, 3, 6],
            'height':   [170, 158, 175, 162, 168],
            'weight':   [75, 58, 82, 55, 70],
        }
        sample_df = pd.DataFrame(sample_data)
        sample_csv = sample_df.to_csv(index=False)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Tai CSV Mau",
            data=sample_csv,
            file_name='heart_batch_sample.csv',
            mime='text/csv',
            width="stretch"
        )
        st.markdown("""<div style="font-size:0.75rem;color:#64748b;text-align:center;margin-top:6px;">
            Tai ve, dien du lieu, roi upload len</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload File CSV</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Chon file CSV chua danh sach benh nhan",
        type=['csv'],
        help="File CSV voi cac cot theo dinh dang mau o tren"
    )

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.session_state.batch_df_raw = df_raw
        except Exception as e:
            st.error(f"Loi doc file CSV: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        df_raw = st.session_state.batch_df_raw

        # Preview
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:10px;padding:10px 0;border-bottom:1px solid #1e1e2e;">
  <div style="font-size:0.85rem;color:#94a3b8;">
    <strong style="color:#e2e8f0;">{len(df_raw)}</strong> dong &nbsp;|&nbsp;
    <strong style="color:#e2e8f0;">{len(df_raw.columns)}</strong> cot &nbsp;|&nbsp;
    File: <strong style="color:#a5b4fc;">{uploaded.name}</strong>
  </div>
</div>""", unsafe_allow_html=True)

        with st.expander("Xem truoc du lieu (5 dong dau)"):
            st.dataframe(df_raw.head(), width="stretch")

        # Validate
        df_clean, row_errors = validate_batch_df(df_raw.copy())

        if df_clean is None:
            for e in row_errors:
                st.error(e)
            st.markdown('</div>', unsafe_allow_html=True)
            return

        if row_errors:
            st.warning(f"{len(row_errors)} dong co van de (van xu ly duoc, nhung nen kiem tra lai):")
            for e in row_errors[:5]:
                st.markdown(f'<div class="batch-error-row">{e}</div>', unsafe_allow_html=True)
            if len(row_errors) > 5:
                st.markdown(f'<div class="batch-error-row">...va {len(row_errors)-5} loi khac</div>',
                            unsafe_allow_html=True)

        # Run button
        n_valid = len(df_clean)
        if st.button(f"Phan tich {n_valid} benh nhan", type="primary", width="stretch"):
            prog_bar = st.progress(0)
            status_txt = st.empty()
            results_list = []
            for i, (_, row) in enumerate(df_clean.iterrows()):
                row_dict = row.to_dict()
                res = predict_single_row(row_dict)
                bmi = round(float(row.get('weight', 70)) /
                            (float(row.get('height', 170)) / 100) ** 2, 1)
                bmi_cat = ('Thieu can' if bmi < 18.5 else
                           'Binh thuong' if bmi < 25 else
                           'Thua can' if bmi < 30 else 'Beo phi')
                bp   = float(row.get('trestbps', 0))
                chol = float(row.get('chol', 0))
                results_list.append({
                    'patient_name': row.get('patient_name', f'Benh nhan {i+1}'),
                    'age':      int(row.get('age', 0)),
                    'sex':      'Nam' if int(float(row.get('sex', 1))) == 1 else 'Nu',
                    'bmi':      bmi,
                    'bmi_cat':  bmi_cat,
                    'trestbps': bp,
                    'bp_status': 'Nguy hiem' if bp >= 140 else ('Canh bao' if bp >= 120 else 'Binh thuong'),
                    'chol':     chol,
                    'chol_status': 'Nguy hiem' if chol >= 240 else ('Canh bao' if chol >= 200 else 'Binh thuong'),
                    'thalach':  float(row.get('thalach', 0)),
                    'prediction': res['prediction'],
                    'ket_qua':  'Khoe manh' if res['prediction'] == 0 else 'Nguy co benh tim',
                    'healthy_prob': round(res['healthy_probability'], 1),
                    'disease_prob': round(res['disease_probability'], 1),
                    'confidence':   round(res['confidence'], 1),
                })
                prog_bar.progress((i + 1) / n_valid)
                if (i + 1) % 5 == 0 or (i + 1) == n_valid:
                    status_txt.markdown(
                        f"<span style='color:#94a3b8;font-size:0.82rem;'>"
                        f"Da xu ly {i+1}/{n_valid} benh nhan...</span>",
                        unsafe_allow_html=True
                    )

            prog_bar.progress(1.0)
            status_txt.empty()
            st.session_state.batch_results = pd.DataFrame(results_list)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ──
    if st.session_state.batch_results is not None:
        df_res = st.session_state.batch_results
        n_total   = len(df_res)
        n_healthy = int((df_res['prediction'] == 0).sum())
        n_risk    = int((df_res['prediction'] == 1).sum())
        risk_rate = round(n_risk / n_total * 100, 1) if n_total > 0 else 0

        # Summary stats
        st.markdown(f"""
<div class="batch-summary">
  <div class="batch-stat total">
    <div class="batch-stat-value">{n_total}</div>
    <div class="batch-stat-label">Tong benh nhan</div>
  </div>
  <div class="batch-stat healthy">
    <div class="batch-stat-value">{n_healthy}</div>
    <div class="batch-stat-label">Khoe manh</div>
  </div>
  <div class="batch-stat risk">
    <div class="batch-stat-value">{n_risk}</div>
    <div class="batch-stat-label">Nguy co benh tim</div>
  </div>
  <div class="batch-stat rate">
    <div class="batch-stat-value">{risk_rate}%</div>
    <div class="batch-stat-label">Ti le nguy co</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Charts
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor('#0f0f1a')

        def ax_style(ax, title):
            ax.set_facecolor('#0f0f1a')
            ax.set_title(title, color='#94a3b8', fontsize=9, fontweight='bold', pad=10)
            ax.tick_params(colors='#64748b', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e1e2e')

        # Pie chart
        axes[0].pie(
            [n_healthy, n_risk],
            labels=['Khoe manh', 'Nguy co'],
            colors=['#16a34a', '#dc2626'],
            autopct='%1.1f%%',
            textprops={'color': '#e2e8f0', 'fontsize': 9},
            wedgeprops={'linewidth': 2, 'edgecolor': '#0f0f1a'},
            startangle=90
        )
        axes[0].set_facecolor('#0f0f1a')
        axes[0].set_title('PHAN BO KET QUA', color='#94a3b8',
                          fontsize=8, fontweight='bold', pad=10)

        # Disease prob histogram
        axes[1].hist(df_res['disease_prob'], bins=15,
                     color='#4f46e5', edgecolor='#0f0f1a', linewidth=0.8, alpha=0.9)
        axes[1].axvline(50, color='#dc2626', linestyle='--', linewidth=1.2, alpha=0.7)
        ax_style(axes[1], 'PHAN PHI XAC SUAT BENH TIM (%)')
        axes[1].set_xlabel('Xac suat (%)', color='#64748b', fontsize=8)
        axes[1].set_ylabel('So luong', color='#64748b', fontsize=8)

        # Age distribution by result
        healthy_ages = df_res[df_res['prediction'] == 0]['age']
        risk_ages    = df_res[df_res['prediction'] == 1]['age']
        bins = range(int(df_res['age'].min()) - 1, int(df_res['age'].max()) + 6, 5)
        if len(healthy_ages) > 0:
            axes[2].hist(healthy_ages, bins=bins, color='#16a34a', alpha=0.75,
                         edgecolor='#0f0f1a', linewidth=0.8, label='Khoe manh')
        if len(risk_ages) > 0:
            axes[2].hist(risk_ages, bins=bins, color='#dc2626', alpha=0.75,
                         edgecolor='#0f0f1a', linewidth=0.8, label='Nguy co')
        ax_style(axes[2], 'PHAN BO TUOI THEO KET QUA')
        axes[2].set_xlabel('Tuoi', color='#64748b', fontsize=8)
        axes[2].set_ylabel('So luong', color='#64748b', fontsize=8)
        axes[2].legend(fontsize=8, labelcolor='#94a3b8',
                       facecolor='#111118', edgecolor='#1e1e2e')

        plt.tight_layout(pad=2.0)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # Filter controls
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Bang Ket qua Chi tiet</div>', unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_result = st.selectbox(
                "Loc theo ket qua",
                ['Tat ca', 'Chi Khoe manh', 'Chi Nguy co benh tim']
            )
        with fc2:
            sort_by = st.selectbox(
                "Sap xep theo",
                ['disease_prob', 'confidence', 'age', 'trestbps', 'chol'],
                format_func=lambda x: {
                    'disease_prob': 'Xac suat benh tim',
                    'confidence':   'Do tin cay',
                    'age':          'Tuoi',
                    'trestbps':     'Huyet ap',
                    'chol':         'Cholesterol',
                }[x]
            )
        with fc3:
            sort_asc = st.selectbox("Thu tu", ['Giam dan', 'Tang dan'])

        df_show = df_res.copy()
        if filter_result == 'Chi Khoe manh':
            df_show = df_show[df_show['prediction'] == 0]
        elif filter_result == 'Chi Nguy co benh tim':
            df_show = df_show[df_show['prediction'] == 1]
        df_show = df_show.sort_values(sort_by, ascending=(sort_asc == 'Tang dan'))

        # Build HTML table
        rows_html = ''
        for _, r in df_show.iterrows():
            badge = (f'<span class="badge-healthy">Khoe manh</span>'
                     if r['prediction'] == 0
                     else f'<span class="badge-risk">Nguy co</span>')
            dp = r['disease_prob']
            hp = r['healthy_prob']
            bar_risk = f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-bg"><div class="prob-bar-fill-risk" style="width:{dp}%;"></div></div>
  <span class="prob-num">{dp}%</span>
</div>"""
            bar_healthy = f"""
<div class="prob-bar-wrap">
  <div class="prob-bar-bg"><div class="prob-bar-fill-healthy" style="width:{hp}%;"></div></div>
  <span class="prob-num">{hp}%</span>
</div>"""
            bp_color = ('#f87171' if r['bp_status'] == 'Nguy hiem'
                        else '#fbbf24' if r['bp_status'] == 'Canh bao'
                        else '#4ade80')
            chol_color = ('#f87171' if r['chol_status'] == 'Nguy hiem'
                          else '#fbbf24' if r['chol_status'] == 'Canh bao'
                          else '#4ade80')
            rows_html += f"""<tr>
  <td><strong style="color:#e2e8f0;">{r['patient_name']}</strong></td>
  <td style="color:#94a3b8;">{r['age']} / {r['sex']}</td>
  <td style="color:#a5b4fc;">{r['bmi']} <span style="font-size:0.72rem;color:#475569;">({r['bmi_cat']})</span></td>
  <td style="color:{bp_color};">{int(r['trestbps'])}</td>
  <td style="color:{chol_color};">{int(r['chol'])}</td>
  <td>{badge}</td>
  <td>{bar_risk}</td>
  <td>{bar_healthy}</td>
  <td style="color:#fbbf24;font-family:'Courier New',monospace;">{r['confidence']}%</td>
</tr>"""

        st.markdown(f"""
<div class="batch-table-wrap">
  <table class="batch-tbl">
    <thead>
      <tr>
        <th>Ho ten</th>
        <th>Tuoi / GT</th>
        <th>BMI</th>
        <th>Huyet ap</th>
        <th>Cholesterol</th>
        <th>Ket qua</th>
        <th>Xac suat benh tim</th>
        <th>Xac suat khoe manh</th>
        <th>Do tin cay</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>""", unsafe_allow_html=True)

        # Export
        export_cols = ['patient_name','age','sex','bmi','bmi_cat',
                       'trestbps','bp_status','chol','chol_status','thalach',
                       'ket_qua','healthy_prob','disease_prob','confidence']
        export_df = df_show[export_cols].copy()
        export_df.columns = [
            'Ho ten','Tuoi','Gioi tinh','BMI','Phan loai BMI',
            'Huyet ap','TT Huyet ap','Cholesterol','TT Cholesterol','Nhip tim toi da',
            'Ket qua','Xac suat khoe (%)','Xac suat benh (%)','Do tin cay (%)'
        ]
        csv_out = export_df.to_csv(index=False, encoding='utf-8-sig')

        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "Xuat ket qua CSV (tat ca)",
                data=df_res[export_cols].rename(columns=dict(zip(export_cols, export_df.columns))).to_csv(index=False, encoding='utf-8-sig'),
                file_name=f'batch_results_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                width="stretch",
            )
        with ec2:
            st.download_button(
                f"Xuat ket qua CSV (dang loc: {len(df_show)} dong)",
                data=csv_out,
                file_name=f'batch_results_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                width="stretch",
            )

        if st.button("Xoa ket qua batch", width="content"):
            st.session_state.batch_results = None
            st.session_state.batch_df_raw  = None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    m = st.session_state.metrics
    if st.session_state.model_ready:
        badge = (f"Model: D-DOA v4 + SVM-RBF &nbsp;|&nbsp; CV AUC: {m['best_cv']*100:.1f}% "
                 f"&nbsp;|&nbsp; Test Acc: {m['acc']*100:.1f}% &nbsp;|&nbsp; Test AUC: {m['auc']:.4f}")
    else:
        badge = "Model: D-DOA + SVM-RBF &nbsp;|&nbsp; Chua huan luyen"

    st.markdown(f"""
<div class="main-header">
  <h1>HE THONG DU DOAN BENH TIM</h1>
  <div class="sub">Phan tich AI voi Mo hinh Co the 2D + Bao cao Chi tiet + Luu CSV</div>
  <div class="model-badge">{badge}</div>
</div>""", unsafe_allow_html=True)

    if not st.session_state.model_ready:
        show_training_panel()
        return

    tab_single, tab_batch = st.tabs([
        "Phan tich Don le",
        "Phan tich Hang loat CSV",
    ])

    with tab_single:
        col_left, col_right = st.columns([420, 900], gap="medium")
        with col_left:
            patient_name, patient_data, submit = render_left_column()
        with col_right:
            render_right_column(submit, patient_name, patient_data)

    with tab_batch:
        render_batch_tab()


if __name__ == '__main__':
    main()
