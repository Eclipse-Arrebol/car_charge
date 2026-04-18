import os

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "map_outputs")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_training_eps40.graphml")
FIG_PATH = os.path.join(OUTPUT_DIR, "guangzhou_tianhe_training_eps40_degree_diagnostics.png")


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    fallback = os.path.basename(path)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(path)


def main():
    graph = ox.load_graphml(_resolve(GRAPH_PATH))

    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree == 0:
            node_colors.append("#d62828")
            node_sizes.append(18)
        elif degree == 1:
            node_colors.append("#f77f00")
            node_sizes.append(9)
        else:
            node_colors.append("#6c757d")
            node_sizes.append(4)

    fig, ax = ox.plot_graph(
        graph,
        node_color=node_colors,
        node_size=node_sizes,
        node_alpha=0.9,
        edge_linewidth=1.2,
        edge_color="#1d3557",
        edge_alpha=0.8,
        bgcolor="white",
        figsize=(14, 14),
        show=False,
        close=False,
    )
    ax.set_title("Guangzhou Tianhe Training Graph (eps=40m, degree diagnostics)", fontsize=15, pad=14)
    fig.savefig(FIG_PATH, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    isolates = list(nx.isolates(graph))
    deg1 = [n for n in graph.nodes() if graph.degree(n) == 1]
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    print(f"Saved figure: {FIG_PATH}")
    print(f"Nodes={graph.number_of_nodes()} Edges={graph.number_of_edges()}")
    print(f"Isolates={len(isolates)} Degree1={len(deg1)} Components={len(components)}")
    if components:
        print(f"Largest component size={len(components[0])}")


if __name__ == "__main__":
    main()
