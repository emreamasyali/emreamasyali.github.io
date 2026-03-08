"""
pages/2_frames.py
=================
Nationalism Frame Analysis page.

Displays frame prevalence, co-occurrence heatmap, and intensity statistics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "results"

st.set_page_config(page_title="Nationalism Frames", page_icon="🏛", layout="wide")

st.title("🏛 Nationalism Frame Analysis")
st.markdown(
    """
Detection of **14 nationalism frames** across the full corpus (~10,326 sentences)
using Llama-3.1-70B. Cross-validated against GPT-4o-mini (κ = 0.76).

Frames are organized across three dimensions: **Identity** (frames 1–7),
**Behavioral** (8–11), and **Relational** (12–14).
"""
)
st.markdown("---")

DIMENSION_COLORS = {
    "Identity": "#2196F3",
    "Behavioral": "#4CAF50",
    "Relational": "#FF9800",
}

FRAME_DIMENSIONS = {
    "State Identification": "Identity",
    "National Identity": "Identity",
    "Territoriality": "Identity",
    "Ancestry": "Identity",
    "Citizenship": "Identity",
    "Language": "Identity",
    "Religion": "Identity",
    "Obedience": "Behavioral",
    "Historical Pride": "Behavioral",
    "Achievements": "Behavioral",
    "Militarism": "Behavioral",
    "Superiority": "Relational",
    "Unity/Diversity": "Relational",
    "Global Mission": "Relational",
}


@st.cache_data
def load_freq():
    path = DATA_DIR / "frame_frequencies.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_cooc():
    path = DATA_DIR / "frame_cooccurrence.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


freq_df = load_freq()
cooc_df = load_cooc()

if freq_df is None:
    st.error("frame_frequencies.csv not found in data/results/. Run notebook 05 first.")
    st.stop()

# Normalise column names
name_col = next((c for c in ["frame_name", "Frame_Name", "name", "Frame"] if c in freq_df.columns), freq_df.columns[0])
count_col = next((c for c in ["count", "Count"] if c in freq_df.columns), freq_df.columns[-1])
pct_col = next((c for c in ["pct_of_sentences", "Percentage", "percentage", "pct"] if c in freq_df.columns), None)

freq_df = freq_df.sort_values(count_col, ascending=False)


# ── Frame Frequency Bar Chart ─────────────────────────────────────────────────

st.markdown("## Frame Prevalence")
col1, col2, col3 = st.columns(3)
col1.metric("Total frames detected", f"{freq_df[count_col].sum():,}")
col2.metric("Most frequent frame", freq_df.iloc[0][name_col])
col3.metric("Avg. frames / nationalist sentence", "1.82")

freq_df["dimension"] = freq_df[name_col].map(FRAME_DIMENSIONS).fillna("Identity")
freq_df["color"] = freq_df["dimension"].map(DIMENSION_COLORS)

if pct_col:
    y_col, y_label = pct_col, "% of corpus sentences"
else:
    y_col, y_label = count_col, "Sentence count"

fig_freq = px.bar(
    freq_df.sort_values(y_col, ascending=True),
    x=y_col,
    y=name_col,
    color="dimension",
    color_discrete_map=DIMENSION_COLORS,
    orientation="h",
    labels={name_col: "", y_col: y_label, "dimension": "Dimension"},
    title="Nationalism Frame Prevalence (Turkish History Textbooks, Grades 9–12)",
    height=500,
)
fig_freq.update_layout(font=dict(size=12), legend_title_text="Dimension")
st.plotly_chart(fig_freq, use_container_width=True)


# ── Co-occurrence Heatmap ──────────────────────────────────────────────────────

st.markdown("## Frame Co-occurrence Heatmap")
st.caption(
    "Shows how often each pair of frames appears in the same sentence. "
    "Normalized by row total (diagonal = 1.0)."
)

if cooc_df is not None and not cooc_df.empty:
    # Normalize by diagonal
    diag = np.diag(cooc_df.values).astype(float)
    diag_safe = np.where(diag == 0, 1.0, diag)
    cooc_norm = cooc_df.values / diag_safe[:, None]
    np.fill_diagonal(cooc_norm, np.nan)  # mask self-co-occurrence

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=cooc_norm,
            x=list(cooc_df.columns),
            y=list(cooc_df.index),
            colorscale="YlOrRd",
            zmin=0,
            zmax=1,
            text=np.round(cooc_norm, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hoverongaps=False,
        )
    )
    fig_heatmap.update_layout(
        title="Frame Co-occurrence (Normalized)",
        height=600,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.info("frame_cooccurrence.csv not found. Run notebook 05 to generate.")


# ── Dimension breakdown ────────────────────────────────────────────────────────

st.markdown("## By Dimension")

dim_totals = freq_df.groupby("dimension")[count_col].sum().reset_index()
fig_pie = px.pie(
    dim_totals,
    names="dimension",
    values=count_col,
    color="dimension",
    color_discrete_map=DIMENSION_COLORS,
    title="Frame Instances by Dimension",
    hole=0.4,
)
fig_pie.update_traces(textinfo="percent+label")

col_pie, col_table = st.columns([1, 1])
with col_pie:
    st.plotly_chart(fig_pie, use_container_width=True)
with col_table:
    st.markdown("### Frame Summary")
    display_cols = [name_col, "dimension", count_col]
    if pct_col:
        display_cols.append(pct_col)
    st.dataframe(freq_df[display_cols].reset_index(drop=True), use_container_width=True)


# ── Methodology note ──────────────────────────────────────────────────────────

with st.expander("Methodology and Validation"):
    st.markdown(
        """
**Model:** Llama-3.1-70B-Instruct-Turbo via TogetherAI API

**Detection approach:** Conservative — model instructed to flag only
explicitly present nationalist discourse, not neutral historical statements.
Multiple frames can be assigned per sentence.

**Cross-validation:** A random 200-sentence sample was independently labeled
using GPT-4o-mini. Cohen's κ = **0.76** (substantial agreement).

**Frame definitions and Turkish examples:** See `prompts/frame_definitions.md`
and `docs/frame_taxonomy.md` in the repository.

**Full prompt:** See `prompts/frame_detection_prompt.md`.
"""
    )
