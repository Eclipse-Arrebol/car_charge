import json
import os
import random
import statistics

import networkx as nx
import osmnx as ox


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
RAW_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_mainroads.graphml")
TRAIN_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_training_stage2.graphml")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_stage2_mapping.json")
REPORT_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_stage2_fidelity_report.json")

SHORT_MAX_M = 1000.0
MEDIUM_MAX_M = 3000.0
SAMPLES_PER_BUCKET = 200
MAX_TRIES = 20000
RANDOM_SEED = 20260418


def _normalize_id(value) -> str:
    return str(value)


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.basename(path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(path)


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
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
        }
    vals = sorted(values)
    return {
        "count": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "median": float(statistics.median(vals)),
        "p95": float(_percentile(vals, 0.95)),
        "p99": float(_percentile(vals, 0.99)),
    }


def _bucket(distance_m: float) -> str:
    if distance_m < SHORT_MAX_M:
        return "short"
    if distance_m <= MEDIUM_MAX_M:
        return "medium"
    return "long"


def main():
    random.seed(RANDOM_SEED)

    raw_graph = ox.load_graphml(_resolve(RAW_GRAPH_PATH))
    train_graph = ox.load_graphml(_resolve(TRAIN_GRAPH_PATH))
    with open(_resolve(MAPPING_PATH), "r", encoding="utf-8") as fh:
        mapping = json.load(fh)

    raw_nodes = list(raw_graph.nodes())
    train_nodes = {_normalize_id(node): node for node in train_graph.nodes()}
    raw_to_train = {
        raw_id: train_nodes[_normalize_id(train_id)]
        for raw_id, train_id in mapping["raw_to_train_node"].items()
        if _normalize_id(train_id) in train_nodes
    }

    cross_results = {"short": [], "medium": [], "long": []}
    same_cluster_distances = {"short": [], "medium": [], "long": []}
    collected = {"short": 0, "medium": 0, "long": 0}
    same_cluster_count = {"short": 0, "medium": 0, "long": 0}
    tries = 0

    while tries < MAX_TRIES and min(collected.values()) < SAMPLES_PER_BUCKET:
        tries += 1
        u, v = random.sample(raw_nodes, 2)
        u_id = _normalize_id(u)
        v_id = _normalize_id(v)
        if u_id not in raw_to_train or v_id not in raw_to_train:
            continue

        try:
            d0 = nx.shortest_path_length(raw_graph, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if d0 <= 1e-6:
            continue

        bucket = _bucket(float(d0))
        if collected[bucket] >= SAMPLES_PER_BUCKET:
            continue

        train_u = raw_to_train[u_id]
        train_v = raw_to_train[v_id]
        if train_u == train_v:
            same_cluster_count[bucket] += 1
            same_cluster_distances[bucket].append(float(d0))
            collected[bucket] += 1
            continue

        try:
            d1 = nx.shortest_path_length(train_graph, train_u, train_v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        err = abs(float(d1) - float(d0)) / float(d0)
        cross_results[bucket].append(err)
        collected[bucket] += 1

    total_counts = {
        bucket: max(1, len(cross_results[bucket]) + same_cluster_count[bucket])
        for bucket in ["short", "medium", "long"]
    }

    report = {
        "config": {
            "short_max_m": SHORT_MAX_M,
            "medium_max_m": MEDIUM_MAX_M,
            "samples_per_bucket": SAMPLES_PER_BUCKET,
            "max_tries": MAX_TRIES,
            "random_seed": RANDOM_SEED,
        },
        "coverage": collected,
        "same_cluster_rate": {
            bucket: same_cluster_count[bucket] / total_counts[bucket]
            for bucket in ["short", "medium", "long"]
        },
        "intra_cluster_stats": {
            bucket: _summary(same_cluster_distances[bucket])
            for bucket in ["short", "medium", "long"]
        },
        "summary": {
            "short": _summary(cross_results["short"]),
            "medium": _summary(cross_results["medium"]),
            "long": _summary(cross_results["long"]),
            "all": _summary(cross_results["short"] + cross_results["medium"] + cross_results["long"]),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("Cross-cluster metrics")
    print("Bucket      Count   Mean     Median   P95      P99")
    for key, label in [("short", "short"), ("medium", "medium"), ("long", "long"), ("all", "all")]:
        item = report["summary"][key]
        print(
            f"{label:<10}{item['count']:<8}"
            f"{(item['mean'] if item['mean'] is not None else float('nan')):<9.4f}"
            f"{(item['median'] if item['median'] is not None else float('nan')):<9.4f}"
            f"{(item['p95'] if item['p95'] is not None else float('nan')):<9.4f}"
            f"{(item['p99'] if item['p99'] is not None else float('nan')):<9.4f}"
        )
    print("\nSame-cluster stats")
    print("Bucket      Rate    MeanD    MedianD  MaxD")
    for key, label in [("short", "short"), ("medium", "medium"), ("long", "long")]:
        item = report["intra_cluster_stats"][key]
        maxd = max(same_cluster_distances[key]) if same_cluster_distances[key] else None
        print(
            f"{label:<10}{report['same_cluster_rate'][key]:<8.4f}"
            f"{(item['mean'] if item['mean'] is not None else float('nan')):<9.2f}"
            f"{(item['median'] if item['median'] is not None else float('nan')):<9.2f}"
            f"{(maxd if maxd is not None else float('nan')):<9.2f}"
        )
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
