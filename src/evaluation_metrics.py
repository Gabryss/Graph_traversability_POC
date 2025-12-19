"""
evaluation_metrics.py

Evaluation utilities for traversability graphs built on top of a traversability map.

Inputs:
- traversability: 2D array [H,W] in [0..1], rock/walls == 0
- nodes_xy: (N,2) float array, nodes in map coords (x,y)
- edges: list[(i,j,weight)] where weight is typically skeleton path length (or any cost)

Outputs:
- metrics dict (JSON-serializable) + optional report printing

Dependencies:
- numpy, networkx (optional but recommended), scipy (optional; improves clearance metric)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import numpy as np

try:
    import networkx as nx
    _NX = True
except Exception:
    _NX = False

try:
    from scipy.spatial import cKDTree
    _KDTREE = True
except Exception:
    _KDTREE = False

try:
    from scipy.ndimage import distance_transform_edt
    _DT = True
except Exception:
    _DT = False


@dataclass
class EvaluationConfig:
    # Sampling stride for “coverage” (distance-to-nearest-node) metric
    coverage_stride: int = 2

    # For line-of-sight (LOS) edge validity test
    los_step: float = 1.0           # sample along segment every N pixels
    obstacle_threshold: float = 0.0 # traversability <= threshold treated as obstacle (0 is typical)

    # Clearance (distance to nearest obstacle) in pixels
    clearance_percentiles: Tuple[float, float, float] = (10.0, 50.0, 90.0)

    # Nearest-neighbor spacing percentiles
    nn_percentiles: Tuple[float, float, float] = (10.0, 50.0, 90.0)

    # Weight statistics (edge weights assumed >=0)
    weight_percentiles: Tuple[float, float, float] = (10.0, 50.0, 90.0)


class GraphEvaluator:
    def __init__(self, cfg: Optional[EvaluationConfig] = None) -> None:
        self.cfg = cfg or EvaluationConfig()

    # ----------------------------
    # Public API
    # ----------------------------
    def evaluate(
        self,
        traversability: np.ndarray,
        nodes_xy: np.ndarray,
        edges: List[Tuple[int, int, float]],
    ) -> Dict[str, Any]:
        if traversability is None or traversability.ndim != 2:
            raise ValueError("traversability must be a 2D numpy array")
        if nodes_xy is None or nodes_xy.ndim != 2 or nodes_xy.shape[1] != 2:
            raise ValueError("nodes_xy must be shape (N,2)")
        if edges is None:
            edges = []

        H, W = traversability.shape
        N = int(nodes_xy.shape[0])
        E = int(len(edges))

        metrics: Dict[str, Any] = {
            "basic": {
                "map_shape": [int(H), int(W)],
                "num_nodes": N,
                "num_edges": E,
                "free_ratio": float(np.mean(traversability > self.cfg.obstacle_threshold)),
            }
        }

        # Node traversability samples
        node_travs = self._sample_node_traversability(traversability, nodes_xy)
        metrics["nodes"] = {
            "trav_mean": float(np.mean(node_travs)) if N else None,
            "trav_min": float(np.min(node_travs)) if N else None,
            "trav_p10_p50_p90": self._percentiles(node_travs, (10, 50, 90)) if N else None,
        }

        # Connectivity / degree stats (NetworkX if available; else fallback)
        if _NX:
            G = self.to_networkx(nodes_xy, edges)
            metrics["graph"] = self._graph_metrics_nx(G)
        else:
            metrics["graph"] = self._graph_metrics_fallback(N, edges)

        # Edge geometry stats
        metrics["edges"] = self._edge_metrics(nodes_xy, edges)

        # Node spacing (nearest neighbor)
        metrics["spacing"] = self._nearest_neighbor_metrics(nodes_xy)

        # Coverage: distance from free pixels to nearest node
        metrics["coverage"] = self._coverage_metrics(traversability, nodes_xy)

        # Clearance: node clearance via distance transform (if scipy available)
        metrics["clearance"] = self._clearance_metrics(traversability, nodes_xy)

        # Edge validity: LOS check over traversability (obstacle crossing)
        metrics["edge_validity"] = self._edge_los_metrics(traversability, nodes_xy, edges)

        return metrics

    def save_json(self, metrics: Dict[str, Any], out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def print_report(self, metrics: Dict[str, Any]) -> None:
        b = metrics.get("basic", {})
        g = metrics.get("graph", {})
        e = metrics.get("edges", {})
        c = metrics.get("coverage", {})
        v = metrics.get("edge_validity", {})
        s = metrics.get("spacing", {})
        cl = metrics.get("clearance", {})

        print("\n========== GRAPH EVALUATION REPORT ==========")
        print(f"- Map shape: {b.get('map_shape')}, free_ratio={b.get('free_ratio'):.3f}")
        print(f"- Nodes: {b.get('num_nodes')} | Edges: {b.get('num_edges')}")
        if g:
            print(f"- Connected components: {g.get('num_components')} | giant_component_ratio={g.get('giant_component_ratio')}")
            print(f"- Avg degree: {g.get('avg_degree')} | degree_p10_p50_p90={g.get('degree_p10_p50_p90')}")
        if e:
            print(f"- Edge euclid_len_p10_p50_p90={e.get('euclid_len_p10_p50_p90')}")
            print(f"- Edge weight_p10_p50_p90={e.get('weight_p10_p50_p90')}")
        if s:
            print(f"- NN dist p10/p50/p90: {s.get('nn_dist_p10_p50_p90')}")
        if c:
            print(f"- Coverage mean_dist_to_node={c.get('mean_dist_to_node')} | p90={c.get('p90_dist_to_node')}")
        if cl:
            print(f"- Node clearance p10/p50/p90: {cl.get('node_clearance_p10_p50_p90')} (pixels)")
        if v:
            print(f"- LOS valid edge ratio: {v.get('valid_ratio')} | invalid_edges={v.get('invalid_edges')}")
        print("============================================\n")

    # ----------------------------
    # Conversions
    # ----------------------------
    def to_networkx(self, nodes_xy: np.ndarray, edges: List[Tuple[int, int, float]]):
        if not _NX:
            raise ImportError("networkx not available")
        G = nx.Graph()
        for i, (x, y) in enumerate(nodes_xy):
            G.add_node(int(i), x=float(x), y=float(y))
        for (i, j, w) in edges:
            ii, jj = int(i), int(j)
            if ii == jj:
                continue
            if ii < 0 or jj < 0 or ii >= len(nodes_xy) or jj >= len(nodes_xy):
                continue
            G.add_edge(ii, jj, weight=float(w))
        return G

    # ----------------------------
    # Metric blocks
    # ----------------------------
    def _sample_node_traversability(self, trav: np.ndarray, nodes_xy: np.ndarray) -> np.ndarray:
        H, W = trav.shape
        if len(nodes_xy) == 0:
            return np.zeros((0,), dtype=float)
        xs = np.clip(np.round(nodes_xy[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(nodes_xy[:, 1]).astype(int), 0, H - 1)
        return trav[ys, xs].astype(float)

    def _graph_metrics_nx(self, G) -> Dict[str, Any]:
        n = G.number_of_nodes()
        if n == 0:
            return {
                "num_components": 0,
                "giant_component_ratio": None,
                "avg_degree": None,
                "degree_p10_p50_p90": None,
            }

        comps = list(nx.connected_components(G))
        comp_sizes = sorted([len(c) for c in comps], reverse=True)
        giant_ratio = float(comp_sizes[0] / n) if comp_sizes else None

        degs = np.array([G.degree[v] for v in G.nodes], dtype=float)
        avg_deg = float(np.mean(degs)) if len(degs) else None

        return {
            "num_components": int(len(comps)),
            "giant_component_ratio": giant_ratio,
            "avg_degree": avg_deg,
            "degree_p10_p50_p90": self._percentiles(degs, (10, 50, 90)) if len(degs) else None,
        }

    def _graph_metrics_fallback(self, N: int, edges: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        # Minimal stats without NX: degree distribution only
        deg = np.zeros((N,), dtype=int)
        for (i, j, _) in edges:
            ii, jj = int(i), int(j)
            if 0 <= ii < N and 0 <= jj < N and ii != jj:
                deg[ii] += 1
                deg[jj] += 1
        degf = deg.astype(float)
        return {
            "num_components": None,
            "giant_component_ratio": None,
            "avg_degree": float(np.mean(degf)) if N else None,
            "degree_p10_p50_p90": self._percentiles(degf, (10, 50, 90)) if N else None,
        }

    def _edge_metrics(self, nodes_xy: np.ndarray, edges: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        if len(edges) == 0 or len(nodes_xy) == 0:
            return {
                "euclid_len_mean": None,
                "euclid_len_p10_p50_p90": None,
                "weight_mean": None,
                "weight_p10_p50_p90": None,
            }

        eu = []
        wts = []
        for (i, j, w) in edges:
            ii, jj = int(i), int(j)
            if ii < 0 or jj < 0 or ii >= len(nodes_xy) or jj >= len(nodes_xy) or ii == jj:
                continue
            dx = float(nodes_xy[ii, 0] - nodes_xy[jj, 0])
            dy = float(nodes_xy[ii, 1] - nodes_xy[jj, 1])
            eu.append(math.hypot(dx, dy))
            wts.append(float(w))

        eu = np.asarray(eu, dtype=float)
        wts = np.asarray(wts, dtype=float)

        return {
            "euclid_len_mean": float(np.mean(eu)) if len(eu) else None,
            "euclid_len_p10_p50_p90": self._percentiles(eu, (10, 50, 90)) if len(eu) else None,
            "weight_mean": float(np.mean(wts)) if len(wts) else None,
            "weight_p10_p50_p90": self._percentiles(wts, self.cfg.weight_percentiles) if len(wts) else None,
        }

    def _nearest_neighbor_metrics(self, nodes_xy: np.ndarray) -> Dict[str, Any]:
        N = len(nodes_xy)
        if N < 2:
            return {"nn_dist_mean": None, "nn_dist_p10_p50_p90": None}

        if not _KDTREE:
            # O(N^2) fallback (fine for small graphs)
            dmins = []
            for i in range(N):
                best = float("inf")
                xi, yi = nodes_xy[i]
                for j in range(N):
                    if i == j:
                        continue
                    dx = float(xi - nodes_xy[j, 0])
                    dy = float(yi - nodes_xy[j, 1])
                    best = min(best, math.hypot(dx, dy))
                dmins.append(best)
            dmins = np.asarray(dmins, dtype=float)
        else:
            tree = cKDTree(nodes_xy)
            d, _ = tree.query(nodes_xy, k=2)  # [self, nn]
            dmins = d[:, 1].astype(float)

        return {
            "nn_dist_mean": float(np.mean(dmins)),
            "nn_dist_p10_p50_p90": self._percentiles(dmins, self.cfg.nn_percentiles),
        }

    def _coverage_metrics(self, trav: np.ndarray, nodes_xy: np.ndarray) -> Dict[str, Any]:
        mask = trav > self.cfg.obstacle_threshold
        free = np.argwhere(mask)  # (y,x)
        if free.size == 0 or len(nodes_xy) == 0:
            return {
                "mean_dist_to_node": None,
                "p90_dist_to_node": None,
                "stride": int(self.cfg.coverage_stride),
            }

        stride = max(1, int(self.cfg.coverage_stride))
        free = free[::stride]
        pts = np.stack([free[:, 1].astype(float), free[:, 0].astype(float)], axis=1)  # (x,y)

        if _KDTREE:
            tree = cKDTree(nodes_xy)
            d, _ = tree.query(pts, k=1)
            d = d.astype(float)
        else:
            # brute force
            d = []
            for (x, y) in pts:
                best = float("inf")
                for (nx_, ny_) in nodes_xy:
                    best = min(best, math.hypot(float(x - nx_), float(y - ny_)))
                d.append(best)
            d = np.asarray(d, dtype=float)

        return {
            "mean_dist_to_node": float(np.mean(d)),
            "p90_dist_to_node": float(np.percentile(d, 90)),
            "stride": int(stride),
            "num_sampled_free_points": int(len(pts)),
        }

    def _clearance_metrics(self, trav: np.ndarray, nodes_xy: np.ndarray) -> Dict[str, Any]:
        if len(nodes_xy) == 0:
            return {"node_clearance_p10_p50_p90": None, "method": None}

        if not _DT:
            return {"node_clearance_p10_p50_p90": None, "method": "distance_transform_unavailable"}

        # distance to obstacle: EDT on free mask
        free = trav > self.cfg.obstacle_threshold
        dist = distance_transform_edt(free.astype(np.uint8)).astype(float)

        H, W = trav.shape
        xs = np.clip(np.round(nodes_xy[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(nodes_xy[:, 1]).astype(int), 0, H - 1)
        node_clear = dist[ys, xs]

        return {
            "node_clearance_p10_p50_p90": self._percentiles(node_clear, self.cfg.clearance_percentiles),
            "method": "scipy.distance_transform_edt",
        }

    def _edge_los_metrics(self, trav: np.ndarray, nodes_xy: np.ndarray, edges: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        if len(edges) == 0 or len(nodes_xy) == 0:
            return {"valid_ratio": None, "invalid_edges": 0, "checked_edges": 0}

        invalid = 0
        checked = 0
        for (i, j, _) in edges:
            ii, jj = int(i), int(j)
            if ii < 0 or jj < 0 or ii >= len(nodes_xy) or jj >= len(nodes_xy) or ii == jj:
                continue
            checked += 1
            if not self._segment_is_free(trav, nodes_xy[ii], nodes_xy[jj]):
                invalid += 1

        valid_ratio = float((checked - invalid) / checked) if checked else None
        return {"valid_ratio": valid_ratio, "invalid_edges": int(invalid), "checked_edges": int(checked)}

    def _segment_is_free(self, trav: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        H, W = trav.shape
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])

        dx = bx - ax
        dy = by - ay
        L = math.hypot(dx, dy)
        if L <= 1e-9:
            return True

        step = max(0.25, float(self.cfg.los_step))
        n = int(math.ceil(L / step))

        for k in range(n + 1):
            t = k / max(1, n)
            x = ax + t * dx
            y = ay + t * dy
            xi = int(np.clip(round(x), 0, W - 1))
            yi = int(np.clip(round(y), 0, H - 1))
            if float(trav[yi, xi]) <= self.cfg.obstacle_threshold:
                return False
        return True

    @staticmethod
    def _percentiles(arr: np.ndarray, ps: Tuple[float, float, float]) -> List[float]:
        a = np.asarray(arr, dtype=float)
        return [float(np.percentile(a, p)) for p in ps]
