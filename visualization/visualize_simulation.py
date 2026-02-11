"""
Animate the environment with a simple heuristic policy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx

from simEvn.Traffic import TrafficPowerEnv


def infer_grid_positions(graph):
    num_nodes = graph.number_of_nodes()
    size = int(np.sqrt(num_nodes))
    if size * size == num_nodes:
        return {i: (i % size, size - 1 - i // size) for i in range(num_nodes)}
    return nx.spring_layout(graph, seed=42)


def choose_actions(env):
    actions = {}
    for ev in env.evs:
        if ev.status == "IDLE" and ev.soc < 30.0:
            # Heuristic: pick station with smaller (queue + price weight)
            best_id = None
            best_score = None
            for s in env.stations:
                score = len(s.queue) + s.current_price * 0.5
                if best_score is None or score < best_score:
                    best_score = score
                    best_id = s.id
            actions[ev.id] = best_id
    return actions


def run_animation(steps=100, interval=600, save_path=None):
    env = TrafficPowerEnv()
    graph = env.traffic_graph
    pos = infer_grid_positions(graph)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_axis_off()

    # Draw edges once
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.4, width=1.0)

    # Station artists
    station_nodes = [s.traffic_node_id for s in env.stations]
    station_sizes = [420 for _ in station_nodes]
    station_colors = [s.current_price for s in env.stations]
    station_scatter = ax.scatter(
        [pos[n][0] for n in station_nodes],
        [pos[n][1] for n in station_nodes],
        s=station_sizes,
        c=station_colors,
        cmap=plt.cm.Oranges,
        marker='s',
        edgecolor='black'
    )

    # EV artists
    ev_scatter = ax.scatter([], [], s=120, c='dodgerblue', edgecolor='white')

    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')
    info = ax.text(0.02, 0.92, "", transform=ax.transAxes, va='top', fontsize=9)

    def init():
        ev_scatter.set_offsets([])
        title.set_text("")
        info.set_text("")
        return ev_scatter, station_scatter, title, info

    def update(frame):
        actions = choose_actions(env)
        _, reward, _, info_dict = env.step(actions)

        ev_positions = np.array([pos[ev.curr_node] for ev in env.evs])
        if ev_positions.size == 0:
            ev_scatter.set_offsets([])
        else:
            ev_scatter.set_offsets(ev_positions)

        # Update stations (size by queue, color by price)
        station_sizes = [420 + len(s.queue) * 220 for s in env.stations]
        station_colors = [s.current_price for s in env.stations]
        station_scatter.set_sizes(station_sizes)
        station_scatter.set_array(np.array(station_colors))

        title.set_text(f"Step {env.time_step:03d} | Reward {reward:.1f}")
        grid_loads = info_dict.get("grid_loads", {})
        info.set_text(f"Grid loads: {grid_loads}")

        return ev_scatter, station_scatter, title, info

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=steps,
        interval=interval,
        blit=False
    )

    if save_path:
        ani.save(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    run_animation(steps=80, interval=700)
