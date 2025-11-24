"""
graph_trav.py

POC for a 2D traversability map + graph built on top of it,
configured via config/config.json.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import networkx as nx
import matplotlib

# Headless-friendly backend (Docker, no X server)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise


class TraversabilityGraphPOC:
    def __init__(
        self,
        width: int,
        height: int,
        perlin_scale: float,
        perlin_octaves: int,
        perlin_seed: int,
        neighbor_mode: str = "4",
    ):
        """
        :param width:  number of cells in X
        :param height: number of cells in Y
        :param perlin_scale: scale factor for Perlin coordinates
        :param perlin_octaves: number of octaves for Perlin noise
        :param perlin_seed: random seed for Perlin noise
        :param neighbor_mode: "4" (von Neumann) or "8" (Moore) connectivity
        """
        self.width = width
        self.height = height
        self.perlin_scale = perlin_scale
        self.perlin_octaves = perlin_octaves
        self.perlin_seed = perlin_seed
        self.neighbor_mode = neighbor_mode

        self.traversability: Optional[np.ndarray] = None
        self.graph: Optional[nx.Graph] = None
        self.path: Optional[List[Tuple[int, int]]] = None

    # ------------------------------------------------------------------ #
    # 1) Map generation
    # ------------------------------------------------------------------ #
    def generate_traversability_map(self) -> np.ndarray:
        """
        Generate a Perlin-noise-based traversability map in [0, 1].

        Returns:
            traversability map as a (height, width) numpy array.
        """
        noise = PerlinNoise(octaves=self.perlin_octaves, seed=self.perlin_seed)
        trav = np.zeros((self.height, self.width), dtype=float)

        for y in range(self.height):
            for x in range(self.width):
                n = noise([x / self.perlin_scale, y / self.perlin_scale])  # [-1, 1]
                trav[y, x] = (n + 1.0) / 2.0  # -> [0, 1]

        self.traversability = trav
        return trav

    # ------------------------------------------------------------------ #
    # 2) Graph construction
    # ------------------------------------------------------------------ #
    def _get_neighbors(self):
        """Return neighbor offsets based on neighbor_mode."""
        if self.neighbor_mode == "4":
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif self.neighbor_mode == "8":
            return [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
        else:
            raise ValueError("neighbor_mode must be '4' or '8'")

    def build_graph(self) -> nx.Graph:
        """
        Build a graph where each cell is a node and edges connect neighbors.

        Node attributes:
            - trav: traversability value in [0, 1]

        Edge attributes:
            - weight: cost = 1 - average_traversability  (lower is better)
        """
        if self.traversability is None:
            raise RuntimeError(
                "Traversability map is not generated. "
                "Call generate_traversability_map() first."
            )

        G = nx.Graph()
        H, W = self.traversability.shape

        # Add nodes
        for y in range(H):
            for x in range(W):
                G.add_node((x, y), trav=self.traversability[y, x])

        # Add edges
        neighbors = self._get_neighbors()
        for y in range(H):
            for x in range(W):
                for dx, dy in neighbors:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < W and 0 <= ny_ < H:
                        t1 = self.traversability[y, x]
                        t2 = self.traversability[ny_, nx_]
                        avg_trav = 0.5 * (t1 + t2)
                        cost = 1.0 - avg_trav  # higher trav -> lower cost
                        G.add_edge((x, y), (nx_, ny_), weight=cost)

        self.graph = G
        return G

    # ------------------------------------------------------------------ #
    # 3) Shortest path
    # ------------------------------------------------------------------ #
    def compute_shortest_path(
        self,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """
        Compute shortest path from start to goal using edge weight 'weight'.

        :param start: (x, y) start cell, default (0, 0)
        :param goal:  (x, y) goal cell, default (width-1, height-1)
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        if start is None:
            start = (0, 0)
        if goal is None:
            goal = (self.width - 1, self.height - 1)

        path = nx.shortest_path(self.graph, source=start, target=goal, weight="weight")
        self.path = path
        return path

    # ------------------------------------------------------------------ #
    # 4) Visualization
    # ------------------------------------------------------------------ #
    def plot(
        self,
        save_path: Optional[Path] = None,
        show: bool = False,
        sample_step: int = 4,
        plot_path: bool = True,
    ):
        """
        Plot the traversability map, a sampled set of graph nodes,
        and optionally the shortest path.

        :param save_path: where to save the figure (PNG, etc.). If None,
                          the figure is not saved.
        :param show:      call plt.show() at the end (only works with GUI).
        :param sample_step: draw only every Nth node for clarity.
        :param plot_path: whether to overlay the stored path.
        """
        if self.traversability is None:
            raise RuntimeError("Traversability map is not generated.")

        H, W = self.traversability.shape

        fig, ax = plt.subplots(figsize=(6, 6))

        # Background: traversability map
        im = ax.imshow(
            self.traversability,
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="Traversability (0=bad, 1=good)")

        # Sampled graph nodes
        xs, ys = np.meshgrid(
            np.arange(0, W, sample_step),
            np.arange(0, H, sample_step),
        )
        ax.scatter(xs, ys, s=2, c="white", alpha=0.3, label="Graph nodes (sampled)")

        # Path overlay
        if plot_path and self.path is not None and len(self.path) > 0:
            path_arr = np.array(self.path)
            ax.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                linewidth=2.0,
                color="red",
                label="Shortest path",
            )

        ax.set_title("Traversability Map + Graph Overlay")
        ax.set_xlabel("x (cell)")
        ax.set_ylabel("y (cell)")
        ax.legend(loc="upper right")
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"[INFO] Saved figure to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------- #
# Config handling & main
# ---------------------------------------------------------------------- #
def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Load JSON config. If config_path is None, assume project structure:

        project_root/
          config/config.json
          src/graph_trav.py

    and compute the path relative to this file.
    """
    if config_path is None:
        # graph_trav.py is in src/
        # project_root = parent of src
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "config" / "config.json"

    with open(config_path, "r") as f:
        cfg = json.load(f)

    return cfg


def main():
    cfg = load_config()

    map_cfg = cfg.get("map", {})
    perlin_cfg = cfg.get("perlin", {})
    graph_cfg = cfg.get("graph", {})
    path_cfg = cfg.get("path", {})
    vis_cfg = cfg.get("visualization", {})

    width = int(map_cfg.get("width", 100))
    height = int(map_cfg.get("height", 100))

    perlin_scale = float(perlin_cfg.get("scale", 30.0))
    perlin_octaves = int(perlin_cfg.get("octaves", 3))
    perlin_seed = int(perlin_cfg.get("seed", 0))

    neighbor_mode = str(graph_cfg.get("neighbor_mode", "4"))

    start_cfg = path_cfg.get("start", [0, 0])
    start = (int(start_cfg[0]), int(start_cfg[1])) if start_cfg is not None else (0, 0)

    goal_cfg = path_cfg.get("goal", None)
    if goal_cfg is None:
        goal: Optional[Tuple[int, int]] = None  # will default to bottom-right
    else:
        goal = (int(goal_cfg[0]), int(goal_cfg[1]))

    sample_step = int(vis_cfg.get("sample_step", 4))
    output_path_str = vis_cfg.get("output_path", "traversability_graph.png")
    show_flag = bool(vis_cfg.get("show", False))

    # Save path is relative to the script's directory (src/)
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / output_path_str

    poc = TraversabilityGraphPOC(
        width=width,
        height=height,
        perlin_scale=perlin_scale,
        perlin_octaves=perlin_octaves,
        perlin_seed=perlin_seed,
        neighbor_mode=neighbor_mode,
    )

    poc.generate_traversability_map()
    poc.build_graph()
    path = poc.compute_shortest_path(start=start, goal=goal)
    print(f"[INFO] Path length: {len(path)} nodes")

    poc.plot(
        save_path=output_path,
        show=show_flag,
        sample_step=sample_step,
        plot_path=True,
    )


if __name__ == "__main__":
    main()

