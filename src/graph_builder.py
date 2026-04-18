"""
Graph Builder
-------------
Constructs a knowledge graph per document from parsed obligation tuples.
Nodes represent entities/obligations, edges represent regulatory relationships.
"""

import json
from pathlib import Path

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import GRAPHS_DIR, GRAPH_FORMAT


def build_graph(parsed_doc: dict) -> nx.DiGraph:
    """Build a directed knowledge graph from parsed obligation tuples."""
    G = nx.DiGraph()
    doc_id = parsed_doc["doc_id"]
    G.graph["doc_id"] = doc_id

    for entry in parsed_doc.get("parsed_tuples", []):
        clause_id = entry["clause_id"]
        t = entry.get("tuple", {})

        # Add obligation node (filter None values — GraphML can't serialize them)
        attrs = {
            "type": "obligation",
            "text": entry.get("text", ""),
            "section": entry.get("section", ""),
        }
        if t.get("obligation") is not None:
            attrs["obligation_desc"] = str(t["obligation"])
        G.add_node(clause_id, **attrs)

        # Add subject node and edge
        subject = t.get("subject")
        if subject:
            subj_id = f"SUBJ:{subject}"
            G.add_node(subj_id, type="subject", label=subject)
            G.add_edge(subj_id, clause_id, relation="has_obligation")

        # Add condition node and edge
        condition = t.get("condition")
        if condition:
            cond_id = f"COND:{clause_id}"
            G.add_node(cond_id, type="condition", label=condition)
            G.add_edge(cond_id, clause_id, relation="triggers")

        # Add threshold node and edge
        threshold = t.get("threshold")
        if threshold:
            thresh_id = f"THRESH:{clause_id}"
            G.add_node(thresh_id, type="threshold", label=str(threshold))
            G.add_edge(clause_id, thresh_id, relation="has_threshold")

        # Add deadline node and edge
        deadline = t.get("deadline")
        if deadline:
            dl_id = f"DL:{clause_id}"
            G.add_node(dl_id, type="deadline", label=str(deadline))
            G.add_edge(clause_id, dl_id, relation="has_deadline")

        # Add exception node and edge
        exception = t.get("exception")
        if exception:
            exc_id = f"EXC:{clause_id}"
            G.add_node(exc_id, type="exception", label=exception)
            G.add_edge(exc_id, clause_id, relation="exempts")

    # Add intra-document edges between obligations sharing subjects
    obligations = [n for n, d in G.nodes(data=True) if d.get("type") == "obligation"]
    subjects = [n for n, d in G.nodes(data=True) if d.get("type") == "subject"]

    for subj in subjects:
        connected_obls = [succ for succ in G.successors(subj)
                          if G.nodes[succ].get("type") == "obligation"]
        for i in range(len(connected_obls)):
            for j in range(i + 1, len(connected_obls)):
                G.add_edge(connected_obls[i], connected_obls[j],
                           relation="co_regulated")

    return G


def _sanitize_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Remove None values from all node/edge attributes (GraphML can't serialize them)."""
    for _, data in G.nodes(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""
    for _, _, data in G.edges(data=True):
        for k in list(data):
            if data[k] is None:
                data[k] = ""
    return G


def save_graph(G: nx.DiGraph, doc_id: str) -> Path:
    """Save graph to disk."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRAPHS_DIR / f"{doc_id}_graph.{GRAPH_FORMAT}"

    _sanitize_graph(G)

    if GRAPH_FORMAT == "graphml":
        nx.write_graphml(G, str(out_path))
    elif GRAPH_FORMAT == "gexf":
        nx.write_gexf(G, str(out_path))
    else:
        # Fallback to JSON adjacency
        data = nx.node_link_data(G)
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(data, f, indent=2)
        out_path = out_path.with_suffix(".json")

    return out_path


def build_and_save(parsed_doc: dict) -> dict:
    """Build graph and save, return summary stats."""
    doc_id = parsed_doc["doc_id"]
    print(f"  [GRAPH] {doc_id}")

    G = build_graph(parsed_doc)
    out_path = save_graph(G, doc_id)

    stats = {
        "doc_id": doc_id,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": {},
        "output_path": str(out_path),
    }

    for _, data in G.nodes(data=True):
        ntype = data.get("type", "unknown")
        stats["node_types"][ntype] = stats["node_types"].get(ntype, 0) + 1

    print(f"    Nodes: {stats['nodes']}, Edges: {stats['edges']}")
    print(f"    Node types: {stats['node_types']}")
    print(f"    Saved to {out_path.name}")

    return stats
