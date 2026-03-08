"""
pages/4_embeddings.py
=====================
Word Embedding Semantic Similarity page.

Displays the category similarity heatmap and explains what high cosine
similarity between ideological categories means in this context.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "results"

st.set_page_config(page_title="Word Embeddings", page_icon="🧠", layout="wide")

st.title("🧠 Word Embedding Similarity")
st.markdown(
    """
Word2Vec (skip-gram, d=100) trained on the full corpus (~146,185 words).
Each of **8 conceptual categories** is represented by a set of Turkish keyword seeds.
Category centroids are computed as mean vectors; pairwise **cosine similarity** reveals
how closely associated the semantic domains are in the textbook vocabulary.

> **Key finding:** Military–Political similarity ≈ **0.9995** — these semantic fields
> are nearly perfectly fused, suggesting that military activity and political power are
> narrated as a single conceptual unit in Turkish history education.
"""
)
st.markdown("---")


@st.cache_data
def load_similarity():
    for path in [
        DATA_DIR / "embedding_similarity.csv",
        DATA_DIR / "word_embeddings_category_similarities.csv",
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0)
                # If it's a long-format file, pivot it
                if df.shape[1] <= 3:
                    return None, df  # long format
                return df, None     # matrix format
            except Exception:
                pass
    return None, None


sim_matrix, sim_long = load_similarity()

# ── Category keyword reference ─────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "Ethnic": ["Türk", "Kürt", "Ermeni", "Rum", "Arap", "Osmanlı", "millet", "ırk"],
    "Political": ["devlet", "hükümet", "iktidar", "meclis", "cumhuriyet", "parti"],
    "Military": ["ordu", "asker", "savaş", "zafer", "sefer", "kuvvet", "komutan"],
    "Cultural": ["kültür", "sanat", "medeniyet", "uygarlık", "mimari", "dil", "kimlik"],
    "Religious": ["din", "İslam", "Müslüman", "Allah", "Hristiyan", "Yahudi", "inanç"],
    "Geographic": ["Anadolu", "Türkiye", "İstanbul", "Ankara", "vatan", "sınır"],
    "Administrative": ["kanun", "anayasa", "hukuk", "mahkeme", "bakanlık", "reform"],
    "Economic": ["ekonomi", "ticaret", "tarım", "sanayi", "para", "vergi", "gelir"],
}


# ── Similarity Heatmap ────────────────────────────────────────────────────────

st.markdown("## Category Similarity Heatmap")

if sim_matrix is not None and not sim_matrix.empty:
    # Full matrix format
    mask = np.eye(len(sim_matrix), dtype=bool)
    display_matrix = sim_matrix.copy()
    display_matrix.values[mask] = np.nan

    fig_heatmap = px.imshow(
        display_matrix,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        text_auto=".3f",
        title="Cosine Similarity Between Conceptual Category Centroids (Word2Vec)",
        height=550,
        aspect="auto",
    )
    fig_heatmap.update_coloraxes(colorbar_title="Cosine<br>Similarity")
    fig_heatmap.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif sim_long is not None and not sim_long.empty:
    # Long format — try to pivot
    cols = sim_long.columns.tolist()
    st.markdown(f"*Columns detected: {cols}*")
    try:
        # Assume: concept/category, target_category, similarity
        if len(cols) >= 3:
            pivot = sim_long.pivot_table(index=cols[0], columns=cols[1], values=cols[2])
            fig_heatmap = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=1,
                text_auto=".3f",
                title="Cosine Similarity (Word2Vec Category Centroids)",
                height=550,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.dataframe(sim_long, use_container_width=True)
    except Exception as e:
        st.error(f"Could not pivot similarity data: {e}")
        st.dataframe(sim_long, use_container_width=True)
else:
    st.info(
        "embedding_similarity.csv not found in data/results/. "
        "Run notebook 03 to generate the similarity matrix."
    )
    # Show placeholder with expected structure
    st.markdown("**Expected category pairs:**")
    cats = list(CATEGORY_KEYWORDS.keys())
    placeholder = pd.DataFrame(
        index=cats, columns=cats,
        data=np.eye(len(cats))
    )
    st.dataframe(placeholder, use_container_width=True)


# ── Interpretation ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
### What does high similarity mean?

A high cosine similarity between two category centroids means that the **words
from both categories occupy nearby regions of the semantic space** — i.e., they
frequently appear in similar contexts in the textbook corpus.

**This matters because:** In a hypothetically neutral historical text, these
categories would be more clearly distinguished. When categories fuse, it
indicates that the textbook consistently narrates one domain *in terms of* another.

**Key patterns in the data:**
- **Military ≈ Political:** Military action is narrated as political achievement
- **Ethnic ≈ Economic:** Ethnic identity discourse overlaps with economic framing
- **Religious isolation:** Islam is treated as a self-contained domain, not integrated
  with political or territorial discourse
"""
    )

with col2:
    st.markdown(
        """
### Category keyword seeds

Each category centroid is computed as the mean of its keyword vectors:
"""
    )
    for cat, keywords in CATEGORY_KEYWORDS.items():
        with st.expander(f"**{cat}** ({len(keywords)} seeds)"):
            st.write(", ".join(keywords))

    st.markdown(
        """
### Model parameters

| Parameter | Value |
|-----------|-------|
| Architecture | Skip-gram |
| Vector size | 100 |
| Window | 5 |
| Min count | 10 |
| Epochs | 15 |
| Subsampling | 1e-4 |
| Negative samples | 10 |
"""
    )


# ── Nearest neighbours explorer ───────────────────────────────────────────────

st.markdown("---")
st.markdown("## Nearest Neighbours Explorer")
st.caption(
    "This feature requires the trained Word2Vec model file (`data/results/word2vec_model.bin`). "
    "If unavailable, run notebook 03 to generate it."
)

model_path = DATA_DIR / "word2vec_model.bin"
if not model_path.exists():
    model_path = ROOT / "results" / "word2vec_model.bin"

if model_path.exists():
    @st.cache_resource
    def load_model():
        from gensim.models import Word2Vec
        return Word2Vec.load(str(model_path))

    try:
        w2v_model = load_model()
        query_word = st.text_input(
            "Enter a Turkish word to find similar words:",
            placeholder="e.g., vatan, savaş, millet",
        )
        if query_word:
            word_lower = query_word.lower().strip()
            if word_lower in w2v_model.wv:
                similar = w2v_model.wv.most_similar(word_lower, topn=15)
                sim_df = pd.DataFrame(similar, columns=["word", "similarity"])

                fig_sim = px.bar(
                    sim_df,
                    x="similarity",
                    y="word",
                    orientation="h",
                    title=f"Words most similar to '{query_word}'",
                    color="similarity",
                    color_continuous_scale="Blues",
                    range_x=[0.5, 1.0],
                    height=400,
                )
                fig_sim.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.warning(f"'{query_word}' not in the model vocabulary. Try another word.")
    except Exception as e:
        st.error(f"Could not load Word2Vec model: {e}")
else:
    st.info("Word2Vec model file not found. Run notebook 03 to generate it.")
