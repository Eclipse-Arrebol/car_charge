import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from map_tools.graph_abstraction import (
        OUTPUT_DIR,
        _edge_length,
        _load_raw_graph,
        _normalize_id,
        build_topology_graph,
        build_training_graph,
    )
else:
    from map_tools.graph_abstraction import (
        OUTPUT_DIR,
        _edge_length,
        _load_raw_graph,
        _normalize_id,
        build_topology_graph,
        build_training_graph,
    )


EPS_VALUES = [15, 20, 25, 30, 35, 40]
SHORT_MAX_M = 1000.0
MEDIUM_MAX_M = 3000.0
SAMPLES_PER_BUCKET = 200
MAX_TRIES = 30000
RANDOM_SEED = 20260418

PAIR_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_scan_pairs.json")
RESULT_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_eps_scan_results.json")
PLOT_A_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_eps_scan_plot_a.png")
PLOT_B_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_eps_scan_plot_b.png")
PLOT_C_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_eps_scan_plot_c.png")


def _bucket(distance_m: float) -> str:
    if distance_m < SHORT_MAX_M:
        return "short"
    if distance_m <= MEDIUM_MAX_M:
        return "medium"
    return "long"


def _percentile(sorted_values, p: float):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "p99": None}
    vals = sorted(values)
    mean = float(sum(vals) / len(vals))
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        median = float((vals[mid - 1] + vals[mid]) / 2.0)
    else:
        median = float(vals[mid])
    return {
        "count": len(vals),
        "mean": mean,
        "median": median,
        "p95": float(_percentile(vals, 0.95)),
        "p99": float(_percentile(vals, 0.99)),
    }


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.basename(path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(path)


def sample_pairs(raw_graph: nx.MultiGraph):
    random.seed(RANDOM_SEED)
    raw_nodes = list(raw_graph.nodes())
    pairs = {"short": [], "medium": [], "long": []}
    tries = 0

    while tries < MAX_TRIES and min(len(v) for v in pairs.values()) < SAMPLES_PER_BUCKET:
        tries += 1
        u, v = random.sample(raw_nodes, 2)
        try:
            d0 = nx.shortest_path_length(raw_graph, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if d0 <= 1e-6:
            continue
        bucket = _bucket(float(d0))
        if len(pairs[bucket]) >= SAMPLES_PER_BUCKET:
            continue
        pairs[bucket].append(
            {"u": _normalize_id(u), "v": _normalize_id(v), "d0": float(d0)}
        )
    return pairs


def save_pairs(pairs):
    with open(PAIR_PATH, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, ensure_ascii=False, indent=2)


def evaluate_eps(raw_graph, training_graph, mapping, pairs):
    train_nodes = {_normalize_id(node): node for node in training_graph.nodes()}
    raw_to_train = {
        raw_id: train_nodes[_normalize_id(train_id)]
        for raw_id, train_id in mapping["raw_to_train_node"].items()
        if _normalize_id(train_id) in train_nodes
    }

    results = {}
    for bucket in ["short", "medium", "long"]:
        cross_errors = []
        same_cluster_distances = []
        no_path_count = 0
        missing_mapping_count = 0
        for item in pairs[bucket]:
            u_id = item["u"]
            v_id = item["v"]
            d0 = item["d0"]
            if u_id not in raw_to_train or v_id not in raw_to_train:
                missing_mapping_count += 1
                continue
            train_u = raw_to_train[u_id]
            train_v = raw_to_train[v_id]
            if train_u == train_v:
                same_cluster_distances.append(float(d0))
                continue
            try:
                d1 = nx.shortest_path_length(training_graph, train_u, train_v, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                no_path_count += 1
                continue
            cross_errors.append(abs(float(d1) - float(d0)) / float(d0))

        total = len(pairs[bucket])
        results[bucket] = {
            "same_cluster_rate": (len(same_cluster_distances) / total) if total else 0.0,
            "no_path_rate": (no_path_count / total) if total else 0.0,
            "missing_mapping_rate": (missing_mapping_count / total) if total else 0.0,
            "cross_cluster_metrics": _summary(cross_errors),
            "intra_cluster_stats": {
                **_summary(same_cluster_distances),
                "max": float(max(same_cluster_distances)) if same_cluster_distances else None,
            },
            "counts": {
                "sampled_pairs": total,
                "cross_cluster_kept": len(cross_errors),
                "same_cluster": len(same_cluster_distances),
                "no_path": no_path_count,
                "missing_mapping": missing_mapping_count,
            },
        }

    edge_rel_errors = []
    raw_lookup = {_normalize_id(node): node for node in raw_graph.nodes()}
    for _, _, _, data in training_graph.edges(keys=True, data=True):
        raw_path = json.loads(data["raw_path"])
        path_nodes = [raw_lookup[node] for node in raw_path if node in raw_lookup]
        if len(path_nodes) < 2:
            continue
        expected = 0.0
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            edge_dict = raw_graph.get_edge_data(a, b, default={})
            if not edge_dict:
                continue
            best_key = min(edge_dict.keys(), key=lambda k: _edge_length(raw_graph, a, b, k))
            expected += _edge_length(raw_graph, a, b, best_key)
        actual = float(data.get("length", 0.0))
        if expected > 1e-9:
            edge_rel_errors.append(abs(actual - expected) / expected)

    components = sorted(nx.connected_components(training_graph), key=len, reverse=True)
    largest_component_size = len(components[0]) if components else 0
    isolates = list(nx.isolates(training_graph))
    low_deg = [n for n in training_graph.nodes() if training_graph.degree(n) == 1]

    return {
        "training_graph_stats": {
            "nodes": training_graph.number_of_nodes(),
            "edges": training_graph.number_of_edges(),
            "compression_ratio": training_graph.number_of_nodes() / max(1, raw_graph.number_of_nodes()),
        },
        "integrity": {
            "component_count": len(components),
            "largest_component_size": largest_component_size,
            "largest_component_ratio": largest_component_size / max(1, training_graph.number_of_nodes()),
            "isolate_count": len(isolates),
            "degree_one_count": len(low_deg),
        },
        "buckets": results,
        "edge_length_error": _summary(edge_rel_errors),
    }


def plot_results(scan_results):
    eps = [item["eps"] for item in scan_results]
    compression = [item["metrics"]["training_graph_stats"]["compression_ratio"] for item in scan_results]
    short_same = [item["metrics"]["buckets"]["short"]["same_cluster_rate"] for item in scan_results]

    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor="white")
    ax1.plot(eps, compression, marker="o", color="#1d3557")
    ax1.set_xlabel("eps (m)")
    ax1.set_ylabel("Node Compression Ratio", color="#1d3557")
    ax1.tick_params(axis="y", labelcolor="#1d3557")
    ax2 = ax1.twinx()
    ax2.plot(eps, short_same, marker="s", color="#d62828")
    ax2.set_ylabel("Short Same-Cluster Rate", color="#d62828")
    ax2.tick_params(axis="y", labelcolor="#d62828")
    ax1.set_title("Compression Ratio vs Short Same-Cluster Rate")
    fig.savefig(PLOT_A_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    for bucket, color in [("short", "#d62828"), ("medium", "#f77f00"), ("long", "#1d3557")]:
        vals = [item["metrics"]["buckets"][bucket]["cross_cluster_metrics"]["p95"] for item in scan_results]
        ax.plot(eps, vals, marker="o", label=bucket, color=color)
    ax.set_xlabel("eps (m)")
    ax.set_ylabel("Cross-Cluster P95 Error")
    ax.set_title("Cross-Cluster P95 vs eps")
    ax.legend()
    fig.savefig(PLOT_B_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    vals = [item["metrics"]["edge_length_error"]["mean"] for item in scan_results]
    ax.plot(eps, vals, marker="o", color="#2a9d8f")
    ax.set_xlabel("eps (m)")
    ax.set_ylabel("Mean Edge Length Relative Error")
    ax.set_title("Edge Length Consistency vs eps")
    fig.savefig(PLOT_C_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_graph = _load_raw_graph()
    topo_graph, *_ = build_topology_graph(raw_graph)
    pairs = sample_pairs(raw_graph)
    save_pairs(pairs)

    scan_results = []
    for eps in EPS_VALUES:
        training_graph, mapping = build_training_graph(raw_graph, topo_graph, tolerance_m=eps)
        metrics = evaluate_eps(raw_graph, training_graph, mapping, pairs)
        scan_results.append({"eps": eps, "metrics": metrics})
        print(
            f"eps={eps:<2} nodes={metrics['training_graph_stats']['nodes']:<4} "
            f"ratio={metrics['training_graph_stats']['compression_ratio']:.3f} "
            f"cc={metrics['integrity']['component_count']:<2} "
            f"iso={metrics['integrity']['isolate_count']:<3} "
            f"short_p95={metrics['buckets']['short']['cross_cluster_metrics']['p95']:.4f} "
            f"med_p95={metrics['buckets']['medium']['cross_cluster_metrics']['p95']:.4f} "
            f"long_p95={metrics['buckets']['long']['cross_cluster_metrics']['p95']:.4f} "
            f"short_same={metrics['buckets']['short']['same_cluster_rate']:.4f} "
            f"short_nopath={metrics['buckets']['short']['no_path_rate']:.4f}"
        )

    result = {
        "config": {
            "eps_values": EPS_VALUES,
            "short_max_m": SHORT_MAX_M,
            "medium_max_m": MEDIUM_MAX_M,
            "samples_per_bucket": SAMPLES_PER_BUCKET,
            "random_seed": RANDOM_SEED,
        },
        "scan_results": scan_results,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    plot_results(scan_results)
    print(f"Saved pair set: {PAIR_PATH}")
    print(f"Saved scan results: {RESULT_PATH}")
    print(f"Saved plots: {PLOT_A_PATH}, {PLOT_B_PATH}, {PLOT_C_PATH}")


if __name__ == "__main__":
    main()
