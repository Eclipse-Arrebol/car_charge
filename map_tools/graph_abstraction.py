import argparse
import json
import os
import shutil
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from osmnx._errors import GraphSimplificationError
from shapely.geometry import LineString


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
RAW_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_mainroads.graphml")
TOPO_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_topology.graphml")
TRAIN_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_training_stage2.graphml")
RESTORED_GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_restored_stage2.graphml")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_stage2_mapping.json")

RAW_PREVIEW_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_raw_stage2.png")
TOPO_PREVIEW_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_topology_stage2.png")
TRAIN_PREVIEW_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_training_stage2.png")
COMPARISON_PREVIEW_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_stage2_comparison.png")

INTERSECTION_TOLERANCE_M = 35
CLUSTER_MAX_PATH_M = 55


def _artifact_paths(tag: Optional[str] = None):
    if not tag:
        return {
            "topology_graph_path": TOPO_GRAPH_PATH,
            "training_graph_path": TRAIN_GRAPH_PATH,
            "restored_graph_path": RESTORED_GRAPH_PATH,
            "mapping_path": MAPPING_PATH,
            "raw_preview_path": RAW_PREVIEW_PATH,
            "topology_preview_path": TOPO_PREVIEW_PATH,
            "training_preview_path": TRAIN_PREVIEW_PATH,
            "comparison_preview_path": COMPARISON_PREVIEW_PATH,
        }

    suffix = f"_{tag}"
    return {
        "topology_graph_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_topology{suffix}.graphml"),
        "training_graph_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_training{suffix}.graphml"),
        "restored_graph_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_restored{suffix}.graphml"),
        "mapping_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe{suffix}_mapping.json"),
        "raw_preview_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_raw{suffix}.png"),
        "topology_preview_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_topology{suffix}.png"),
        "training_preview_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe_training{suffix}.png"),
        "comparison_preview_path": os.path.join(OUTPUT_DIR, f"guangzhou_tianhe{suffix}_comparison.png"),
    }


def _resolve_existing_path(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.basename(path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(path)


def _safe_display_path(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


def _normalize_id(value) -> str:
    return str(value)


def _load_raw_graph(path: str = RAW_GRAPH_PATH) -> nx.MultiGraph:
    graph = ox.load_graphml(_resolve_existing_path(path))
    if not isinstance(graph, nx.MultiGraph):
        graph = nx.MultiGraph(graph)
    graph.graph.setdefault("crs", "epsg:4326")
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component).copy()


def _edge_length(graph: nx.MultiGraph, u, v, key) -> float:
    data = graph.get_edge_data(u, v, key, default={})
    try:
        return float(data.get("length", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _node_point(graph: nx.MultiGraph, node):
    attrs = graph.nodes[node]
    return (float(attrs.get("x", 0.0)), float(attrs.get("y", 0.0)))


def _path_geometry(graph: nx.MultiGraph, path_nodes):
    coords = []
    for idx, node in enumerate(path_nodes):
        point = _node_point(graph, node)
        if idx == 0 or point != coords[-1]:
            coords.append(point)
    if len(coords) == 1:
        coords.append(coords[0])
    return LineString(coords)


def _best_edge_key(graph: nx.MultiGraph, u, v):
    edge_dict = graph.get_edge_data(u, v, default={})
    if not edge_dict:
        return None
    return min(edge_dict.keys(), key=lambda k: _edge_length(graph, u, v, k))


def _path_length_from_nodes(graph: nx.MultiGraph, path_nodes) -> float:
    total = 0.0
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        best_key = _best_edge_key(graph, u, v)
        if best_key is None:
            continue
        total += _edge_length(graph, u, v, best_key)
    return total


def _edge_sequence_from_nodes(graph: nx.MultiGraph, path_nodes):
    sequence = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        best_key = _best_edge_key(graph, u, v)
        if best_key is None:
            continue
        sequence.append([_normalize_id(u), _normalize_id(v), int(best_key)])
    return sequence


def _compose_paths(*paths):
    merged = []
    for path in paths:
        if not path:
            continue
        if not merged:
            merged.extend(path)
            continue
        if merged[-1] == path[0]:
            merged.extend(path[1:])
        else:
            merged.extend(path)
    return merged


def _cluster_internal_path(raw_graph: nx.MultiGraph, raw_nodes, source, target):
    if source == target:
        return [source]
    allowed = [node for node in raw_nodes if node in raw_graph.nodes()]
    if source not in allowed:
        allowed.append(source)
    if target not in allowed:
        allowed.append(target)
    subgraph = raw_graph.subgraph(allowed)
    try:
        return nx.shortest_path(subgraph, source, target, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        try:
            return nx.shortest_path(raw_graph, source, target, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [source, target]


def _cluster_representative(topo_graph: nx.MultiGraph, topo_nodes):
    topo_nodes = list(topo_nodes)
    if len(topo_nodes) == 1:
        return topo_nodes[0]
    xs = [float(topo_graph.nodes[node].get("x", 0.0)) for node in topo_nodes]
    ys = [float(topo_graph.nodes[node].get("y", 0.0)) for node in topo_nodes]
    cx = sum(xs) / max(1, len(xs))
    cy = sum(ys) / max(1, len(ys))

    def rank(node):
        degree = topo_graph.degree(node)
        x = float(topo_graph.nodes[node].get("x", 0.0))
        y = float(topo_graph.nodes[node].get("y", 0.0))
        dist2 = (x - cx) ** 2 + (y - cy) ** 2
        return (-degree, dist2, int(node))

    return min(topo_nodes, key=rank)


def build_topology_graph(raw_graph: nx.MultiGraph, preserve_nodes: Optional[set] = None):
    preserve_nodes = preserve_nodes or set()
    raw_directed = nx.MultiDiGraph(raw_graph)
    raw_directed.graph.update(raw_graph.graph)
    for node in raw_directed.nodes():
        raw_directed.nodes[node]["raw_nodes"] = json.dumps([_normalize_id(node)], ensure_ascii=True)
        raw_directed.nodes[node]["is_anchor"] = _normalize_id(node) in preserve_nodes

    try:
        topo_directed = ox.simplify_graph(
            raw_directed,
            node_attrs_include=["is_anchor"],
            remove_rings=False,
            track_merged=True,
        )
        topo_graph = ox.convert.to_undirected(topo_directed)
        largest_component = max(nx.connected_components(topo_graph), key=len)
        topo_graph = topo_graph.subgraph(largest_component).copy()

        raw_to_topo = {}
        topo_to_raw = {}
        edge_mapping = []

        for topo_node, data in topo_graph.nodes(data=True):
            raw_nodes = json.loads(data.get("raw_nodes", "[]"))
            topo_to_raw[str(topo_node)] = raw_nodes
            for raw_node in raw_nodes:
                raw_to_topo[raw_node] = int(topo_node)

        for topo_u, topo_v, _, data in topo_graph.edges(keys=True, data=True):
            raw_path = json.loads(data.get("raw_path", "[]")) if "raw_path" in data else []
            if not raw_path:
                raw_u = topo_to_raw[str(topo_u)][0]
                raw_v = topo_to_raw[str(topo_v)][0]
                raw_path = [raw_u, raw_v]
            raw_edge_sequence = json.loads(data.get("raw_edge_sequence", "[]")) if "raw_edge_sequence" in data else []
            edge_mapping.append(
                {
                    "topo_u": int(topo_u),
                    "topo_v": int(topo_v),
                    "raw_path": raw_path,
                    "raw_edge_sequence": raw_edge_sequence,
                }
            )
        simplified = True
    except GraphSimplificationError:
        topo_graph = raw_graph.copy()
        raw_to_topo = {}
        topo_to_raw = {}
        edge_mapping = []
        for idx, node in enumerate(topo_graph.nodes()):
            attrs = topo_graph.nodes[node]
            attrs["raw_nodes"] = json.dumps([_normalize_id(node)], ensure_ascii=True)
            attrs["raw_anchor"] = _normalize_id(node)
            raw_to_topo[_normalize_id(node)] = idx
            topo_to_raw[str(idx)] = [_normalize_id(node)]
        mapping = {node: idx for idx, node in enumerate(list(topo_graph.nodes()))}
        topo_graph = nx.relabel_nodes(topo_graph, mapping)
        for u, v, key, data in topo_graph.edges(keys=True, data=True):
            raw_u = topo_to_raw[str(u)][0]
            raw_v = topo_to_raw[str(v)][0]
            edge_mapping.append(
                {
                    "topo_u": int(u),
                    "topo_v": int(v),
                    "raw_path": [raw_u, raw_v],
                    "raw_edge_sequence": [[raw_u, raw_v, int(key)]],
                }
            )
        simplified = False

    return topo_graph, raw_to_topo, topo_to_raw, edge_mapping, simplified


def _cluster_topology_nodes(topo_graph: nx.MultiGraph, tolerance_m: float, max_path_m: float):
    projected = ox.project_graph(topo_graph)
    positions = {
        node: (float(data.get("x", 0.0)), float(data.get("y", 0.0)))
        for node, data in projected.nodes(data=True)
    }
    nodes = list(projected.nodes())
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from(nodes)

    for idx, u in enumerate(nodes):
        ux, uy = positions[u]
        for v in nodes[idx + 1:]:
            vx, vy = positions[v]
            dx = ux - vx
            dy = uy - vy
            euclidean = (dx * dx + dy * dy) ** 0.5
            if euclidean > tolerance_m:
                continue
            try:
                path_len = nx.shortest_path_length(projected, u, v, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if path_len <= max_path_m:
                cluster_graph.add_edge(u, v)

    components = list(nx.connected_components(cluster_graph))
    topo_to_cluster = {}
    cluster_to_topo = {}
    cluster_attrs = {}

    for cluster_id, comp in enumerate(components):
        topo_nodes = sorted(int(node) for node in comp)
        cluster_to_topo[str(cluster_id)] = topo_nodes
        representative = _cluster_representative(topo_graph, topo_nodes)
        rep_attrs = topo_graph.nodes[representative]
        street_counts = [int(topo_graph.nodes[node].get("street_count", 0)) for node in comp]
        cluster_attrs[cluster_id] = {
            "x": float(rep_attrs.get("x", 0.0)),
            "y": float(rep_attrs.get("y", 0.0)),
            "street_count": max(street_counts) if street_counts else 0,
            "representative_topo_node": int(representative),
        }
        for topo_node in topo_nodes:
            topo_to_cluster[str(topo_node)] = cluster_id

    return topo_to_cluster, cluster_to_topo, cluster_attrs


def build_training_graph(raw_graph: nx.MultiGraph, topo_graph: nx.MultiGraph, tolerance_m: float, output_paths=None):
    output_paths = output_paths or _artifact_paths()
    raw_lookup = {_normalize_id(node): node for node in raw_graph.nodes()}
    topo_to_cluster, cluster_to_topo, cluster_attrs = _cluster_topology_nodes(
        topo_graph, tolerance_m=tolerance_m, max_path_m=CLUSTER_MAX_PATH_M
    )

    training_graph = nx.MultiGraph()
    training_graph.graph.update(raw_graph.graph)

    raw_to_train = {}
    train_to_raw = {}
    cluster_representatives = {}
    for cluster_id, topo_nodes in cluster_to_topo.items():
        raw_nodes = []
        for topo_node in topo_nodes:
            raw_nodes.extend(json.loads(topo_graph.nodes[topo_node]["raw_nodes"]))
        raw_nodes = sorted(set(raw_nodes))
        attrs = cluster_attrs[int(cluster_id)]
        representative_topo = attrs["representative_topo_node"]
        representative_raw = json.loads(topo_graph.nodes[representative_topo]["raw_nodes"])[0]
        training_graph.add_node(
            int(cluster_id),
            x=attrs["x"],
            y=attrs["y"],
            street_count=attrs["street_count"],
            raw_nodes=json.dumps(raw_nodes, ensure_ascii=True),
            cluster_size=len(raw_nodes),
            representative_topo_node=int(representative_topo),
            representative_raw_node=_normalize_id(representative_raw),
        )
        train_to_raw[str(cluster_id)] = raw_nodes
        cluster_representatives[int(cluster_id)] = {
            "topo_node": int(representative_topo),
            "raw_node": _normalize_id(representative_raw),
        }
        for raw_node in raw_nodes:
            raw_to_train[raw_node] = int(cluster_id)

    pair_to_segments = defaultdict(list)
    for topo_u, topo_v, _, data in topo_graph.edges(keys=True, data=True):
        cluster_u = topo_to_cluster[str(topo_u)]
        cluster_v = topo_to_cluster[str(topo_v)]
        if cluster_u == cluster_v:
            continue
        key = (cluster_u, cluster_v) if cluster_u <= cluster_v else (cluster_v, cluster_u)
        raw_path = json.loads(data["raw_path"]) if "raw_path" in data else [
            topo_graph.nodes[topo_u]["raw_anchor"],
            topo_graph.nodes[topo_v]["raw_anchor"],
        ]
        raw_edge_sequence = json.loads(data["raw_edge_sequence"]) if "raw_edge_sequence" in data else []
        if not raw_edge_sequence:
            edge_dict = topo_graph.get_edge_data(topo_u, topo_v, default={})
            edge_key = int(next(iter(edge_dict.keys()))) if edge_dict else 0
            raw_edge_sequence = [[raw_path[0], raw_path[1], edge_key]]
        pair_to_segments[key].append(
            {
                "topo_u": int(topo_u),
                "topo_v": int(topo_v),
                "length": float(data.get("length", 0.0)),
                "raw_path": raw_path,
                "raw_edge_sequence": raw_edge_sequence,
            }
        )

    edge_mapping = []
    for (cluster_u, cluster_v), segments in pair_to_segments.items():
        best = min(segments, key=lambda item: item["length"])
        rep_u = raw_lookup[cluster_representatives[int(cluster_u)]["raw_node"]]
        rep_v = raw_lookup[cluster_representatives[int(cluster_v)]["raw_node"]]
        base_path_nodes = [raw_lookup[node] for node in best["raw_path"] if node in raw_lookup]
        if len(base_path_nodes) < 2:
            continue
        prefix = _cluster_internal_path(raw_graph, train_to_raw[str(cluster_u)], rep_u, base_path_nodes[0])
        suffix = _cluster_internal_path(raw_graph, train_to_raw[str(cluster_v)], base_path_nodes[-1], rep_v)
        full_path_nodes = _compose_paths(prefix, base_path_nodes, suffix)
        if len(full_path_nodes) < 2:
            continue
        geometry = _path_geometry(raw_graph, full_path_nodes)
        training_graph.add_edge(
            cluster_u,
            cluster_v,
            length=best["length"],
            geometry=geometry,
            raw_path=json.dumps(best["raw_path"], ensure_ascii=True),
            raw_edge_sequence=json.dumps(best["raw_edge_sequence"], ensure_ascii=True),
            collapsed_segment_count=len(segments),
        )
        edge_mapping.append(
            {
                "train_u": int(cluster_u),
                "train_v": int(cluster_v),
                "raw_path": best["raw_path"],
                "raw_edge_sequence": best["raw_edge_sequence"],
                "collapsed_segment_count": len(segments),
            }
        )

    largest_component = max(nx.connected_components(training_graph), key=len)
    training_graph = training_graph.subgraph(largest_component).copy()

    edge_mapping = [
        item for item in edge_mapping
        if item["train_u"] in training_graph.nodes() and item["train_v"] in training_graph.nodes()
    ]

    mapping = {
        "raw_graph_path": RAW_GRAPH_PATH,
        "topology_graph_path": output_paths["topology_graph_path"],
        "training_graph_path": output_paths["training_graph_path"],
        "restored_graph_path": output_paths["restored_graph_path"],
        "intersection_tolerance_m": tolerance_m,
        "cluster_max_path_m": CLUSTER_MAX_PATH_M,
        "raw_graph_stats": {
            "nodes": raw_graph.number_of_nodes(),
            "edges": raw_graph.number_of_edges(),
        },
        "topology_graph_stats": {
            "nodes": topo_graph.number_of_nodes(),
            "edges": topo_graph.number_of_edges(),
        },
        "training_graph_stats": {
            "nodes": training_graph.number_of_nodes(),
            "edges": training_graph.number_of_edges(),
        },
        "raw_to_train_node": raw_to_train,
        "train_to_raw_nodes": train_to_raw,
        "train_representative_nodes": cluster_representatives,
        "edge_mapping": edge_mapping,
    }
    return training_graph, mapping


def save_mapping(mapping, path: str = MAPPING_PATH):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=True, indent=2)


def render_graph(graph, output_path: str, title: str, node_size: float, edge_width: float):
    fig, ax = ox.plot_graph(
        graph,
        node_size=node_size,
        node_color="#d62828",
        node_alpha=0.85,
        edge_linewidth=edge_width,
        edge_color="#1d3557",
        edge_alpha=0.75,
        bgcolor="white",
        figsize=(14, 14),
        show=False,
        close=False,
    )
    ax.set_title(title, fontsize=15, pad=14)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_comparison(raw_graph, topo_graph, train_graph, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor="white")
    panels = [
        (raw_graph, "L0 Raw Drivable Roads", 5, 0.7),
        (topo_graph, "Stage2 Topology Graph", 8, 0.8),
        (train_graph, "Stage2 Training Graph", 9, 0.9),
    ]
    for ax, (graph, title, node_size, edge_width) in zip(axes, panels):
        ox.plot_graph(
            graph,
            node_size=node_size,
            node_color="#d62828",
            node_alpha=0.85,
            edge_linewidth=edge_width,
            edge_color="#1d3557",
            edge_alpha=0.75,
            bgcolor="white",
            show=False,
            close=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=12)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build training graph abstraction from raw Tianhe road graph.")
    parser.add_argument("--eps", type=float, default=INTERSECTION_TOLERANCE_M, help="Clustering tolerance in meters.")
    parser.add_argument("--tag", type=str, default=None, help="Optional artifact tag suffix.")
    return parser.parse_args()


def main():
    args = _parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    auto_tag = None
    if args.tag:
        auto_tag = args.tag
    elif abs(args.eps - INTERSECTION_TOLERANCE_M) > 1e-9:
        eps_label = str(int(args.eps)) if float(args.eps).is_integer() else str(args.eps).replace(".", "_")
        auto_tag = f"eps{eps_label}"
    output_paths = _artifact_paths(auto_tag)

    raw_graph = _load_raw_graph()
    print(
        f"Loaded raw graph: nodes={raw_graph.number_of_nodes()}, "
        f"edges={raw_graph.number_of_edges()}"
    )

    topo_graph, raw_to_topo, topo_to_raw, topo_edge_mapping, topology_simplified = build_topology_graph(raw_graph)
    print(
        f"Built topology graph: nodes={topo_graph.number_of_nodes()}, "
        f"edges={topo_graph.number_of_edges()}"
    )

    training_graph, mapping = build_training_graph(raw_graph, topo_graph, args.eps, output_paths=output_paths)
    mapping["raw_to_topology_node"] = raw_to_topo
    mapping["topology_to_raw_nodes"] = topo_to_raw
    mapping["topology_edge_mapping"] = topo_edge_mapping
    mapping["topology_simplified"] = topology_simplified
    print(
        f"Built stage2 training graph: nodes={training_graph.number_of_nodes()}, "
        f"edges={training_graph.number_of_edges()}"
    )

    ox.save_graphml(topo_graph, output_paths["topology_graph_path"])
    ox.save_graphml(training_graph, output_paths["training_graph_path"])
    shutil.copyfile(_resolve_existing_path(RAW_GRAPH_PATH), output_paths["restored_graph_path"])
    save_mapping(mapping, output_paths["mapping_path"])

    render_graph(raw_graph, output_paths["raw_preview_path"], "Guangzhou Tianhe Raw Drivable Roads", 5, 0.7)
    render_graph(topo_graph, output_paths["topology_preview_path"], "Guangzhou Tianhe Topology Graph", 8, 0.8)
    render_graph(
        training_graph,
        output_paths["training_preview_path"],
        f"Guangzhou Tianhe Stage2 Training Graph ({args.eps:g}m clustering)",
        9,
        0.9,
    )
    render_comparison(raw_graph, topo_graph, training_graph, output_paths["comparison_preview_path"])

    print(f"Saved topology graph: {_safe_display_path(output_paths['topology_graph_path'])}")
    print(f"Saved stage2 training graph: {_safe_display_path(output_paths['training_graph_path'])}")
    print(f"Saved stage2 mapping: {_safe_display_path(output_paths['mapping_path'])}")
    print(f"Saved comparison preview: {_safe_display_path(output_paths['comparison_preview_path'])}")


if __name__ == "__main__":
    main()
