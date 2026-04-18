"""
Graph Aligner
-------------
Cross-document graph alignment using semantic similarity.
Identifies equivalent or related obligation nodes across regulatory documents.
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
    GRAPHS_DIR, GRAPH_FORMAT, OUTPUTS_DIR,
)


def load_graphs(graph_dir: Path = GRAPHS_DIR) -> dict[str, nx.DiGraph]:
    """Load all saved graphs."""
    graphs = {}
    for gfile in sorted(graph_dir.glob(f"*_graph.{GRAPH_FORMAT}")):
        doc_id = gfile.stem.replace("_graph", "")
        if doc_id == "merged":
            continue  # Skip previously generated merged graphs
        if GRAPH_FORMAT == "graphml":
            G = nx.read_graphml(str(gfile))
        elif GRAPH_FORMAT == "gexf":
            G = nx.read_gexf(str(gfile))
        else:
            with open(gfile) as f:
                data = json.load(f)
            G = nx.node_link_graph(data)
        graphs[doc_id] = G
    return graphs


def get_obligation_texts(G: nx.DiGraph) -> dict[str, str]:
    """Extract obligation node IDs and their text."""
    return {
        node: data.get("text", data.get("obligation_desc", ""))
        for node, data in G.nodes(data=True)
        if data.get("type") == "obligation" and data.get("text")
    }


def compute_cross_doc_alignments(graphs: dict[str, nx.DiGraph]) -> list[dict]:
    """Compute pairwise obligation alignments across all document pairs."""
    print("  Loading sentence transformer...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Collect all obligation texts with doc_id prefix
    all_obligations = []
    for doc_id, G in graphs.items():
        texts = get_obligation_texts(G)
        for node_id, text in texts.items():
            all_obligations.append({
                "doc_id": doc_id,
                "node_id": node_id,
                "text": text,
            })

    if len(all_obligations) < 2:
        print("  [WARN] Not enough obligations for cross-doc alignment")
        return []

    # Group by document for vectorized cross-doc computation
    obls_by_doc = {}
    for o in all_obligations:
        obls_by_doc.setdefault(o["doc_id"], []).append(o)

    doc_ids = list(obls_by_doc.keys())
    total_obls = sum(len(v) for v in obls_by_doc.values())
    print(f"  Computing embeddings for {total_obls} obligations...")

    # Encode per document
    embeddings_by_doc = {}
    for doc_id, obls in obls_by_doc.items():
        embeddings_by_doc[doc_id] = model.encode(
            [o["text"] for o in obls], normalize_embeddings=True, show_progress_bar=False
        )

    # Vectorized cross-doc similarity via matrix multiplication
    alignments = []
    for di in range(len(doc_ids)):
        for dj in range(di + 1, len(doc_ids)):
            da, db = doc_ids[di], doc_ids[dj]
            emb_a, emb_b = embeddings_by_doc[da], embeddings_by_doc[db]
            sim_matrix = emb_a @ emb_b.T
            rows, cols = np.where(sim_matrix >= SIMILARITY_THRESHOLD)
            obls_a, obls_b = obls_by_doc[da], obls_by_doc[db]
            for r, c in zip(rows, cols):
                alignments.append({
                    "doc_a": da,
                    "node_a": obls_a[r]["node_id"],
                    "text_a": obls_a[r]["text"][:200],
                    "doc_b": db,
                    "node_b": obls_b[c]["node_id"],
                    "text_b": obls_b[c]["text"][:200],
                    "similarity": round(float(sim_matrix[r, c]), 4),
                })

    alignments.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"  Found {len(alignments)} cross-document alignments (threshold={SIMILARITY_THRESHOLD})")

    return alignments


def build_merged_graph(graphs: dict[str, nx.DiGraph], alignments: list[dict]) -> nx.DiGraph:
    """Build a merged multi-document graph with alignment edges."""
    merged = nx.DiGraph()

    # Add all nodes/edges from individual graphs with doc_id prefix
    for doc_id, G in graphs.items():
        for node, data in G.nodes(data=True):
            node_attrs = {k: v for k, v in data.items() if k != "doc_id"}
            merged.add_node(f"{doc_id}::{node}", doc_id=doc_id, **node_attrs)
        for u, v, data in G.edges(data=True):
            merged.add_edge(f"{doc_id}::{u}", f"{doc_id}::{v}", **data)

    # Add alignment edges
    for align in alignments:
        merged.add_edge(
            f"{align['doc_a']}::{align['node_a']}",
            f"{align['doc_b']}::{align['node_b']}",
            relation="cross_doc_alignment",
            similarity=align["similarity"],
        )

    return merged


def run_alignment() -> dict:
    """Run the full cross-document alignment pipeline."""
    print("\n=== Cross-Document Graph Alignment ===")

    graphs = load_graphs()
    if len(graphs) < 2:
        print("  [WARN] Need at least 2 document graphs for alignment")
        return {"alignments": [], "merged_graph_stats": {}}

    print(f"  Loaded {len(graphs)} graphs: {list(graphs.keys())}")

    alignments = compute_cross_doc_alignments(graphs)
    merged = build_merged_graph(graphs, alignments)

    # Save merged graph (sanitize None values for GraphML)
    merged_path = GRAPHS_DIR / f"merged_graph.{GRAPH_FORMAT}"
    for _, data in merged.nodes(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""
    for _, _, data in merged.edges(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""
    if GRAPH_FORMAT == "graphml":
        nx.write_graphml(merged, str(merged_path))
    print(f"  Merged graph: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    print(f"  Saved to {merged_path.name}")

    # Save alignments
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    align_path = OUTPUTS_DIR / "reports" / "cross_doc_alignments.json"
    align_path.parent.mkdir(parents=True, exist_ok=True)
    with open(align_path, "w") as f:
        json.dump(alignments, f, indent=2)

    return {
        "num_alignments": len(alignments),
        "merged_nodes": merged.number_of_nodes(),
        "merged_edges": merged.number_of_edges(),
        "alignments": alignments[:10],  # Top 10 for summary
    }
