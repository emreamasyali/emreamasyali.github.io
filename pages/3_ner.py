"""
pages/3_ner.py
==============
Named Entity Recognition and co-occurrence network explorer page.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "results"

st.set_page_config(page_title="Named Entities", page_icon="🔍", layout="wide")

st.title("🔍 Named Entity Recognition")
st.markdown(
    """
Entity extraction using [`savasy/bert-base-turkish-ner-cased`](https://huggingface.co/savasy/bert-base-turkish-ner-cased),
a Turkish BERT model fine-tuned for NER. Results from processing **10,326 sentences**
with batch size 20 and confidence threshold 0.6.

- **34,077** total entity mentions | **8,728** unique entities
- **LOC** 2,181 unique | **ORG** 4,733 unique | **PER** 1,775 unique
"""
)
st.markdown("---")

TYPE_COLORS = {"LOC": "#2196F3", "ORG": "#4CAF50", "PER": "#FF9800"}


@st.cache_data
def load_ner_summary():
    # Try results directory first, then sample
    for path in [
        DATA_DIR / "ner_entities_summary.csv",
        DATA_DIR / "all_entities_summary.csv",
        ROOT / "data" / "sample" / "sample_ner_output.csv",
    ]:
        if path.exists():
            return pd.read_csv(path), str(path)
    return None, None


@st.cache_data
def load_cooccurrence():
    path = DATA_DIR / "cooccurrence_matrix.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


ner_df, source_path = load_ner_summary()
cooc_df = load_cooccurrence()

if ner_df is None:
    st.error("No NER summary CSV found in data/results/. Run notebook 02 first.")
    st.stop()

# Detect columns
entity_col = next((c for c in ["entity_text", "Entity", "entity"] if c in ner_df.columns), ner_df.columns[0])
type_col = next((c for c in ["entity_type", "Type", "type"] if c in ner_df.columns), None)
count_col = next((c for c in ["mention_count", "count", "Count", "frequency"] if c in ner_df.columns), None)

if source_path:
    st.caption(f"Data source: {Path(source_path).name}")


# ── Sidebar filters ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")
    top_n = st.slider("Top N entities per type", 5, 30, 15)
    if type_col and count_col:
        min_count = st.slider("Minimum mentions", 1, 50, 3)
    else:
        min_count = 1


# ── Top entities per type ─────────────────────────────────────────────────────

st.markdown("## Top Entities by Type")

if type_col and count_col:
    entity_types = ner_df[type_col].unique()
    cols = st.columns(min(3, len(entity_types)))
    for i, etype in enumerate(sorted(entity_types)):
        with cols[i % len(cols)]:
            top = (
                ner_df[ner_df[type_col] == etype]
                .nlargest(top_n, count_col)
            )
            color = TYPE_COLORS.get(etype, "#9E9E9E")
            fig = px.bar(
                top,
                x=count_col,
                y=entity_col,
                orientation="h",
                title=f"Top {top_n} {etype} Entities",
                color_discrete_sequence=[color],
                labels={count_col: "Mentions", entity_col: ""},
                height=450,
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                font=dict(size=11),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    # Simple table fallback
    st.dataframe(ner_df.head(50), use_container_width=True)


# ── Co-occurrence Network ──────────────────────────────────────────────────────

st.markdown("## Entity Co-occurrence Network")
st.caption(
    "Entities that appear in the same sentence are connected by an edge. "
    "Node size = degree centrality. Adjust the minimum co-occurrence threshold below."
)

if cooc_df is not None and not cooc_df.empty:
    ent_a_col = cooc_df.columns[0]
    ent_b_col = cooc_df.columns[1]
    cooc_count_col = cooc_df.columns[2] if len(cooc_df.columns) > 2 else None

    min_cooc = st.slider("Minimum co-occurrence count", 2, 50, 10)

    if cooc_count_col:
        edges = cooc_df[cooc_df[cooc_count_col] >= min_cooc]
    else:
        edges = cooc_df

    if len(edges) == 0:
        st.info("No pairs meet the minimum threshold. Lower the slider.")
    else:
        # Build networkx graph for layout
        try:
            import networkx as nx
            G = nx.Graph()
            for _, row in edges.iterrows():
                w = row[cooc_count_col] if cooc_count_col else 1
                G.add_edge(str(row[ent_a_col]), str(row[ent_b_col]), weight=float(w))

            if G.number_of_nodes() > 100:
                st.warning(f"Large network ({G.number_of_nodes()} nodes). Showing top 80 by degree.")
                top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:80]
                G = G.subgraph([n for n, _ in top_nodes]).copy()

            pos = nx.spring_layout(G, k=1.8, seed=42, weight="weight")

            # Build Plotly traces
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.6, color="#CCCCCC"),
                hoverinfo="none",
            )

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_degree = [G.degree(n) for n in G.nodes()]
            node_labels = list(G.nodes())

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo="text",
                marker=dict(
                    size=[5 + d * 2 for d in node_degree],
                    color=node_degree,
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Degree"),
                ),
            )

            fig_net = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Entity Co-occurrence Network (min. {min_cooc} shared sentences)",
                    showlegend=False,
                    hovermode="closest",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    margin=dict(l=10, r=10, t=40, b=10),
                ),
            )
            st.plotly_chart(fig_net, use_container_width=True)
            st.caption(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        except ImportError:
            st.info("Install `networkx` for network visualization. Showing table instead.")
            st.dataframe(edges.head(100), use_container_width=True)
else:
    st.info(
        "cooccurrence_matrix.csv not found in data/results/. "
        "Run notebook 02 to generate co-occurrence data."
    )


# ── Entity frequency table ─────────────────────────────────────────────────────

st.markdown("## Entity Frequency Table")
search = st.text_input("Search entities", placeholder="e.g., Osmanlı, Mustafa, Anadolu")

disp = ner_df.copy()
if count_col:
    disp = disp.sort_values(count_col, ascending=False)
if min_count and count_col:
    disp = disp[disp[count_col] >= min_count]
if search:
    disp = disp[disp[entity_col].str.contains(search, case=False, na=False)]

st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=400)
st.caption(f"Showing {len(disp):,} entities.")
