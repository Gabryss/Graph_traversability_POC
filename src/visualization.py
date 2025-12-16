"""
visualization.py

Centralized visualization utilities for:
- Traversability maps (2D grid in [0,1])
- Coarse traversability graph overlay (nodes + edges)
- Optional cluster/cell debug overlays

This file should contain ALL plotting logic, so environment.py and graph_build.py
remain "pure" generation/build modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Iterable, Any

import numpy as np
import matplotlib

# Headless-friendly backend (Docker / CI / no X server)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TraversabilityVisualizer:
    def __init__(
        self,
        map_cmap: str = "viridis",
        node_cmap: str = "plasma",
        map_vmin: float = 0.0,
        map_vmax: float = 1.0,
        node_vmin: float = 0.0,
        node_vmax: float = 1.0,
    ):
        self.map_cmap = map_cmap
        self.node_cmap = node_cmap
        self.map_vmin = map_vmin
        self.map_vmax = map_vmax
        self.node_vmin = node_vmin
        self.node_vmax = node_vmax

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _finalize_figure(
        fig: Any,
        save_path: Optional[Path],
        show: bool,
        dpi: int = 200,
    ) -> None:
        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi)
            print(f"[INFO] Saved figure to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------ #
    # 1) Traversability map
    # ------------------------------------------------------------------ #
    def plot_traversability_map(
        self,
        traversability: np.ndarray,
        title: str = "Traversability Map",
        save_path: Optional[Path] = None,
        show: bool = False,
        figsize: Tuple[float, float] = (6, 6),
        interpolation: str = "nearest",
    ) -> None:
        if traversability is None:
            raise ValueError("traversability is None")
        if traversability.ndim != 2:
            raise ValueError("traversability must be a 2D array")

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            traversability,
            origin="lower",
            cmap=self.map_cmap,
            interpolation=interpolation,
            vmin=self.map_vmin,
            vmax=self.map_vmax,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Traversability (0 = rock, 1 = best)")

        ax.set_title(title)
        ax.set_xlabel("x (cell)")
        ax.set_ylabel("y (cell)")

        self._finalize_figure(fig, save_path, show)

    # ------------------------------------------------------------------ #
    # 2) Graph overlay (nodes + edges)
    # ------------------------------------------------------------------ #
    def plot_graph_overlay(
        self,
        traversability: np.ndarray,
        graph: Any,
        title: str = "Coarse Graph (nodes + edges)",
        save_path: Optional[Path] = None,
        show: bool = False,
        figsize: Tuple[float, float] = (6, 6),
        interpolation: str = "nearest",
        # nodes
        node_size: float = 20.0,
        node_edgecolor: str = "black",
        node_linewidth: float = 0.4,
        node_alpha: float = 0.95,
        # edges
        edge_color: str = "white",
        edge_width: float = 1.0,
        edge_alpha: float = 0.4,
        # labels (optional)
        draw_node_ids: bool = False,
        node_id_fontsize: int = 7,
        node_id_color: str = "white",
    ) -> None:
        """
        Expects nodes with:
          - data["center"] = (cx, cy) in cell coordinates
          - data["trav"]   = node traversability in [0,1]
        """
        if traversability is None:
            raise ValueError("traversability is None")
        if graph is None:
            raise ValueError("graph is None")

        fig, ax = plt.subplots(figsize=figsize)

        # Background map
        im = ax.imshow(
            traversability,
            origin="lower",
            cmap=self.map_cmap,
            interpolation=interpolation,
            vmin=self.map_vmin,
            vmax=self.map_vmax,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Traversability (0 = rock, 1 = best)")

        # Edges
        for (u, v) in graph.edges():
            cx1, cy1 = graph.nodes[u]["center"]
            cx2, cy2 = graph.nodes[v]["center"]
            ax.plot(
                [cx1, cx2],
                [cy1, cy2],
                color=edge_color,
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=2,
            )

        # Nodes
        xs, ys, colors = [], [], []
        for nid, data in graph.nodes(data=True):
            cx, cy = data["center"]
            xs.append(cx)
            ys.append(cy)
            colors.append(data.get("trav", 0.0))

            if draw_node_ids:
                ax.text(
                    cx,
                    cy,
                    str(nid),
                    fontsize=node_id_fontsize,
                    color=node_id_color,
                    ha="center",
                    va="center",
                    zorder=4,
                )

        sc = ax.scatter(
            xs,
            ys,
            s=node_size,
            c=colors,
            cmap=self.node_cmap,
            edgecolors=node_edgecolor,
            linewidths=node_linewidth,
            alpha=node_alpha,
            zorder=3,
            vmin=self.node_vmin,
            vmax=self.node_vmax,
        )
        cbar_nodes = fig.colorbar(sc, ax=ax)
        cbar_nodes.set_label("Node avg traversability")

        ax.set_title(title)
        ax.set_xlabel("x (cell)")
        ax.set_ylabel("y (cell)")

        self._finalize_figure(fig, save_path, show)

    # ------------------------------------------------------------------ #
    # 3) Optional debug: draw cluster cell footprints
    # ------------------------------------------------------------------ #
    def plot_graph_clusters(
        self,
        traversability: np.ndarray,
        graph: Any,
        title: str = "Coarse Graph (cluster cells)",
        save_path: Optional[Path] = None,
        show: bool = False,
        figsize: Tuple[float, float] = (6, 6),
        cell_marker_size: float = 8.0,
        cell_alpha: float = 0.35,
        draw_centers: bool = True,
        center_size: float = 25.0,
    ) -> None:
        """
        Expects node attribute:
          - data["cells"]  = list[(x,y)] cluster footprint in fine grid coords
          - data["center"] = (cx,cy)
        """
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            traversability,
            origin="lower",
            cmap=self.map_cmap,
            interpolation="nearest",
            vmin=self.map_vmin,
            vmax=self.map_vmax,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Traversability (0 = rock, 1 = best)")

        # Draw each node's cell footprint
        for _, data in graph.nodes(data=True):
            cells = data.get("cells", [])
            if not cells:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            ax.scatter(xs, ys, s=cell_marker_size, alpha=cell_alpha, zorder=2)

        # Optionally draw centers on top
        if draw_centers:
            xs, ys = [], []
            for _, data in graph.nodes(data=True):
                cx, cy = data["center"]
                xs.append(cx)
                ys.append(cy)
            ax.scatter(xs, ys, s=center_size, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("x (cell)")
        ax.set_ylabel("y (cell)")

        self._finalize_figure(fig, save_path, show)
