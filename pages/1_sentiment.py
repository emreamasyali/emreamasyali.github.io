"""
pages/1_sentiment.py
====================
Sentiment Analysis explorer page for the Turkish Textbook NLP dashboard.

Shows sentiment distribution (POSITIVE/NEGATIVE/NEUTRAL) for each
ethnic/religious group, with interactive group filtering.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "results"

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")

st.title("🎭 Sentiment Analysis")
st.markdown(
    """
Sentiment classification of **2,361 sentences** referencing 7 ethnic and religious
groups, using Llama-3.1-70B (TogetherAI API). Validated against human annotations
(82.6% accuracy, κ = 0.708).

**Label scheme:**
- **POSITIVE** — group portrayed favorably, heroically, or as sympathetic victims
- **NEGATIVE** — group portrayed as threats, enemies, or aggressors
- **NEUTRAL** — factual, objective description
"""
)
st.markdown("---")


# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    path = DATA_DIR / "sentiment_by_group.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


df = load_data()

if df is None:
    st.error(
        "sentiment_by_group.csv not found in data/results/. "
        "Please run notebook 04 first or check the data directory."
    )
    st.stop()

# Detect column names flexibly
group_col = next((c for c in ["group", "Group"] if c in df.columns), df.columns[0])
label_col = next(
    (c for c in ["predicted_label", "Label", "label", "sentiment"] if c in df.columns),
    df.columns[1]
)
# Normalise labels to uppercase
df[label_col] = df[label_col].str.upper().str.strip()

ALL_GROUPS = sorted(df[group_col].unique())
COLORS = {
    "POSITIVE": "#4CAF50",
    "NEGATIVE": "#F44336",
    "NEUTRAL": "#9E9E9E",
}


# ── Sidebar filters ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    selected_groups = st.multiselect(
        "Select groups",
        options=ALL_GROUPS,
        default=ALL_GROUPS,
        help="Filter by ethnic or religious group",
    )
    show_neutral = st.checkbox("Show NEUTRAL sentiment", value=True)

if not selected_groups:
    st.warning("Please select at least one group.")
    st.stop()

filtered = df[df[group_col].isin(selected_groups)]
if not show_neutral:
    filtered = filtered[filtered[label_col] != "NEUTRAL"]


# ── Aggregated distribution ────────────────────────────────────────────────────

st.markdown("## Sentiment Distribution by Group")

pivot = (
    filtered.groupby([group_col, label_col])
    .size()
    .reset_index(name="count")
)
total_per_group = pivot.groupby(group_col)["count"].transform("sum")
pivot["percentage"] = pivot["count"] / total_per_group * 100

labels_order = [l for l in ["POSITIVE", "NEGATIVE", "NEUTRAL"] if l in pivot[label_col].unique()]

fig_bar = px.bar(
    pivot,
    x=group_col,
    y="percentage",
    color=label_col,
    barmode="stack",
    color_discrete_map=COLORS,
    category_orders={label_col: labels_order},
    labels={group_col: "Group", "percentage": "% of sentences", label_col: "Sentiment"},
    title="Sentiment Distribution by Ethnic/Religious Group",
    height=450,
)
fig_bar.update_layout(
    legend_title_text="Sentiment",
    yaxis_range=[0, 100],
    xaxis_title="",
    font=dict(size=13),
)
st.plotly_chart(fig_bar, use_container_width=True)


# ── Positive / Negative ratio ──────────────────────────────────────────────────

st.markdown("## Positive / Negative Ratio")
st.caption("Ratio > 1 means more positive sentences; ratio < 1 means more negative sentences.")

ratio_data = (
    pivot[pivot[label_col].isin(["POSITIVE", "NEGATIVE"])]
    .pivot_table(index=group_col, columns=label_col, values="count", fill_value=0)
    .reset_index()
)
for col in ["POSITIVE", "NEGATIVE"]:
    if col not in ratio_data.columns:
        ratio_data[col] = 0

ratio_data["ratio"] = ratio_data["POSITIVE"] / ratio_data["NEGATIVE"].replace(0, 0.01)
ratio_data["ratio_label"] = ratio_data["ratio"].apply(lambda x: f"{x:.2f}x")

fig_ratio = go.Figure(go.Bar(
    x=ratio_data[group_col],
    y=ratio_data["ratio"],
    text=ratio_data["ratio_label"],
    textposition="outside",
    marker_color=[
        COLORS["POSITIVE"] if r >= 1 else COLORS["NEGATIVE"]
        for r in ratio_data["ratio"]
    ],
))
fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="gray",
                    annotation_text="Balanced (1.0)", annotation_position="right")
fig_ratio.update_layout(
    title="Positive-to-Negative Ratio by Group",
    xaxis_title="",
    yaxis_title="Positive / Negative ratio",
    height=380,
    showlegend=False,
)
st.plotly_chart(fig_ratio, use_container_width=True)


# ── Data table ─────────────────────────────────────────────────────────────────

st.markdown("## Detailed Data")

search_term = st.text_input("Search sentences", placeholder="Type to filter...")

display_df = filtered.copy()
if search_term:
    text_cols = [c for c in display_df.columns if display_df[c].dtype == object]
    mask = display_df[text_cols].apply(
        lambda col: col.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    display_df = display_df[mask]

st.dataframe(display_df, use_container_width=True, height=400)
st.caption(f"Showing {len(display_df):,} of {len(filtered):,} sentences.")

# Download
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", csv, "sentiment_filtered.csv", "text/csv")
