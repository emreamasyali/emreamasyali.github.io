"""
streamlit_app.py
================
Main entry point for the Turkish Textbook NLP dashboard.

Runs entirely from pre-computed CSV files in data/results/.
No API keys or corpus required.

Run locally:
    streamlit run app/streamlit_app.py

Deploy on HuggingFace Spaces:
    - Set app file to app/streamlit_app.py
    - Include data/results/ in the repo
    - Use requirements-app.txt
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make data/ accessible from the app/ subdirectory
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "results"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turkish Textbook NLP",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data
def load_sentiment():
    path = DATA_DIR / "sentiment_by_group.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_frames():
    path = DATA_DIR / "frame_frequencies.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_frame_cooccurrence():
    path = DATA_DIR / "frame_cooccurrence.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        return df
    return pd.DataFrame()


@st.cache_data
def load_cooccurrence():
    path = DATA_DIR / "cooccurrence_matrix.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_embeddings():
    path = DATA_DIR / "embedding_similarity.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Python-3.13-blue.svg",
        use_container_width=False,
    )
    st.markdown("## Turkish Textbook NLP")
    st.markdown(
        "Computational analysis of nationalist discourse in Turkish secondary "
        "education history textbooks (Grades 9–12, 2024–25 curriculum)."
    )
    st.markdown("---")
    st.markdown("**Author:** Emre Amasyali")
    st.markdown("**Institution:** Institut Barcelona d'Estudis Internacionals")
    st.markdown("---")
    st.markdown(
        "**Source code:** [GitHub](https://github.com/emreamasyali/nlpturkishtextbooks)"
    )
    st.markdown(
        "**Notebooks:** "
        "[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/emreamasyali/nlpturkishtextbooks/"
        "blob/main/notebooks/01_pdf_to_text.ipynb)"
    )
    st.markdown("---")
    st.markdown(
        "Use the **pages** in the sidebar above to explore each analysis:\n"
        "- 🎭 Sentiment Analysis\n"
        "- 🏛 Nationalism Frames\n"
        "- 🔍 Named Entities\n"
        "- 🧠 Word Embeddings"
    )


# ── Main page ─────────────────────────────────────────────────────────────────

st.title("📚 Nationalist Discourse in Turkish History Textbooks")
st.subheader("A Computational Text Analysis")

st.markdown(
    """
This dashboard presents findings from a computational NLP study of four Turkish
secondary education history textbooks (Grades 9–12, 2024–25 curriculum), issued
by the Turkish Ministry of National Education (MEB).

The analysis covers four methodological stages: Named Entity Recognition, Word
Embeddings, LLM-based Sentiment Analysis, and Nationalism Frame Detection.
Navigate using the sidebar pages to explore each component interactively.
"""
)

st.markdown("---")

# ── Key statistics ────────────────────────────────────────────────────────────

st.markdown("## Corpus Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Textbooks", "4", help="Grades 9–12, MEB 2024–25 curriculum")

with col2:
    st.metric("Words (cleaned)", "~146,185")

with col3:
    st.metric("Sentences", "~10,326")

with col4:
    st.metric("Sentences analyzed\n(sentiment)", "2,361")

st.markdown("---")

# ── Pipeline overview ─────────────────────────────────────────────────────────

st.markdown("## Pipeline Overview")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        """
**Stage 1 — Named Entity Recognition**
- Model: `savasy/bert-base-turkish-ner-cased`
- 34,077 total entity mentions
- 8,728 unique entities (PER / LOC / ORG)
- Top entity: *Osmanlı Devleti* (1,040 mentions)
- Top person: *Mustafa Kemal* (152 mentions)

**Stage 2 — Word Embeddings**
- Gensim Word2Vec, skip-gram
- Vector size: 100, Window: 5, Epochs: 15
- 8 conceptual categories analyzed
- Military–Political similarity: **0.9995**
"""
    )

with col_b:
    st.markdown(
        """
**Stage 3 — Sentiment Analysis**
- Model: Llama-3.1-70B (TogetherAI)
- 2,361 sentences across 7 groups
- Human validation: accuracy **82.6%** (κ = 0.708)
- Cross-validated with GPT-4

**Stage 4 — Nationalism Frame Detection**
- 14 nationalism frames (4 dimensions)
- Model: Llama-3.1-70B
- Cross-validated with GPT-4o-mini (κ = 0.76)
- Historical Pride most frequent (8.7% of sentences)
- Average **1.82 frames** per nationalist sentence
"""
    )

st.markdown("---")

# ── Quick summary charts ──────────────────────────────────────────────────────

st.markdown("## Quick Overview")

sentiment_df = load_sentiment()
frames_df = load_frames()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sentiment by Group (preview)")
    if not sentiment_df.empty:
        # Try to show a simple group × label summary
        label_col = None
        for candidate in ["predicted_label", "Label", "label", "sentiment"]:
            if candidate in sentiment_df.columns:
                label_col = candidate
                break

        group_col = None
        for candidate in ["group", "Group"]:
            if candidate in sentiment_df.columns:
                group_col = candidate
                break

        if label_col and group_col:
            pivot = (
                sentiment_df.groupby([group_col, label_col])
                .size()
                .unstack(fill_value=0)
            )
            for col in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["POSITIVE", "NEGATIVE", "NEUTRAL"]]
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
            st.dataframe(pivot_pct.round(1).style.format("{:.1f}%"), use_container_width=True)
        else:
            st.dataframe(sentiment_df.head(10), use_container_width=True)
    else:
        st.info("Sentiment data not found in data/results/. Run notebook 04 first.")

with col2:
    st.markdown("### Top Nationalism Frames (preview)")
    if not frames_df.empty:
        name_col = next((c for c in ["frame_name", "Frame_Name", "name"] if c in frames_df.columns), None)
        count_col = next((c for c in ["count", "Count"] if c in frames_df.columns), None)
        if name_col and count_col:
            top = frames_df[[name_col, count_col]].sort_values(count_col, ascending=False).head(8)
            st.bar_chart(top.set_index(name_col))
        else:
            st.dataframe(frames_df.head(8), use_container_width=True)
    else:
        st.info("Frame frequency data not found in data/results/. Run notebook 05 first.")

st.markdown("---")
st.markdown(
    "_Data source: MEB Turkish history textbooks (Grades 9–12, 2024–25). "
    "Raw PDFs not included — see [data/README.md](data/README.md) for access instructions._"
)
