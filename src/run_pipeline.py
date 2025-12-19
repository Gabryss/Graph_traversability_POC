"""
run_pipeline.py

Orchestrates the whole project:
1) Load config.json
2) Generate or load environment (traversability map)
3) Build graph (via graph_build.py)
4) Evaluate graph (evaluation_metrics.py)
5) Save artifacts + plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from environment import EnvironmentGenerator
from evaluation_metrics import EvaluationConfig, GraphEvaluator
from graph_build import GraphBuildConfig, build_graph_from_traversability
from visualization import TraversabilityVisualizer


EdgeTuple = Tuple[int, int, float]  # legacy (u,v,weight)


# ----------------------------
# Config helpers
# ----------------------------
def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)


def infer_project_root(config_path: Path) -> Path:
    """
    Robust project root inference:
    - If config path is project_root/config/config.json -> root is parent of 'config'
    - Else root is config_path.parent
    """
    config_path = config_path.resolve()
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def build_graph_build_cfg(cfg: Dict[str, Any]) -> GraphBuildConfig:
    gbc = GraphBuildConfig()

    block = cfg.get("graph_build", {})
    if not isinstance(block, dict):
        block = {}

    for k, v in block.items():
        if hasattr(gbc, k):
            setattr(gbc, k, v)

    # Default map_path from visualization.map_npy_path if not provided
    vis = cfg.get("visualization", {})
    if isinstance(vis, dict):
        map_npy = vis.get("map_npy_path")
        if map_npy and not block.get("map_path"):
            gbc.map_path = str(map_npy)

    return gbc


def build_eval_cfg(cfg: Dict[str, Any]) -> EvaluationConfig:
    ec = EvaluationConfig()
    block = cfg.get("evaluation", {})
    if not isinstance(block, dict):
        block = {}

    for k, v in block.items():
        if hasattr(ec, k):
            setattr(ec, k, v)

    # Optional mapping from validation / graph
    val = cfg.get("validation", {})
    if isinstance(val, dict) and "los_step" in val:
        ec.los_step = float(val["los_step"])

    graph = cfg.get("graph", {})
    if isinstance(graph, dict) and "obstacle_threshold" in graph:
        ec.obstacle_threshold = float(graph["obstacle_threshold"])

    return ec


def resolve_under_root(project_root: Path, p: Union[str, Path]) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (project_root / p).resolve()


# ----------------------------
# Environment stage
# ----------------------------
def run_environment(cfg: Dict[str, Any], project_root: Path) -> np.ndarray:
    map_cfg = cfg.get("map", {})
    perlin_cfg = cfg.get("perlin", {})
    cave_cfg = cfg.get("cave", {})
    vis_cfg = cfg.get("visualization", {})

    width = int(map_cfg.get("width", 100))
    height = int(map_cfg.get("height", 100))
    generator = str(map_cfg.get("generator", "cave")).lower()
    use_existing = bool(map_cfg.get("use_existing", True))

    perlin_scale = float(perlin_cfg.get("scale", 30.0))
    perlin_octaves = int(perlin_cfg.get("octaves", 3))
    perlin_seed = int(perlin_cfg.get("seed", 0))

    cave_fill_prob = float(cave_cfg.get("fill_probability", 0.45))
    cave_birth_limit = int(cave_cfg.get("birth_limit", 4))
    cave_death_limit = int(cave_cfg.get("death_limit", 3))
    cave_steps = int(cave_cfg.get("steps", 5))
    cave_min_trav = float(cave_cfg.get("min_traversability", 0.3))  # optional

    map_npy_path = resolve_under_root(project_root, vis_cfg.get("map_npy_path", "data/traversability_map.npy"))
    env_png_path = resolve_under_root(project_root, vis_cfg.get("env_output_path", "data/traversability_map.png"))
    show = bool(vis_cfg.get("show", False))

    if use_existing and map_npy_path.exists():
        print(f"[INFO] Using existing traversability map: {map_npy_path}")
        trav = np.load(map_npy_path)
    else:
        if use_existing:
            print(f"[WARN] map not found at {map_npy_path}; generating new one.")
        else:
            print("[INFO] Forced environment regeneration (use_existing=false).")

        env = EnvironmentGenerator(
            width=width,
            height=height,
            map_generator=generator,
            perlin_scale=perlin_scale,
            perlin_octaves=perlin_octaves,
            perlin_seed=perlin_seed,
            cave_fill_prob=cave_fill_prob,
            cave_birth_limit=cave_birth_limit,
            cave_death_limit=cave_death_limit,
            cave_steps=cave_steps,
            cave_min_traversability=cave_min_trav,
        )
        trav = env.generate_traversability_map()

        map_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_npy_path, trav)
        print(f"[INFO] Saved traversability map to: {map_npy_path}")

    viz = TraversabilityVisualizer()
    viz.plot_traversability_map(
        traversability=trav,
        title=f"Traversability Map ({generator})",
        save_path=env_png_path,
        show=show,
    )
    return trav


# ----------------------------
# Graph stage
# ----------------------------
def run_graph(cfg: Dict[str, Any], project_root: Path):
    gb_cfg = build_graph_build_cfg(cfg)

    gb_cfg.map_path = str(resolve_under_root(project_root, gb_cfg.map_path))
    gb_cfg.output_dir = str(resolve_under_root(project_root, gb_cfg.output_dir))

    print(f"[INFO] GraphBuildConfig: {asdict(gb_cfg)}")
    nodes_xy, edges = build_graph_from_traversability(gb_cfg)
    return nodes_xy, edges, Path(gb_cfg.output_dir)


def load_graph_artifacts(out_dir: Path) -> Tuple[np.ndarray, List[Any]]:
    """
    Backward compatible loader:
    - nodes: graph_nodes.npy
    - edges: prefers graph_edges.npz (routed), falls back to graph_edges.npy
    """
    out_dir = Path(out_dir)
    npath = out_dir / "graph_nodes.npy"
    if not npath.exists():
        raise FileNotFoundError(f"Missing {npath}")
    nodes_xy = np.load(npath)

    # New format: NPZ
    ep_npz = out_dir / "graph_edges.npz"
    if ep_npz.exists():
        z = np.load(ep_npz, allow_pickle=True)
        u = z["u"].astype(int)
        v = z["v"].astype(int)
        w = z["w"].astype(float)
        paths = z["paths"]  # object array of int arrays

        # Build lightweight edge objects (dicts) so downstream can handle routed edges
        edges = []
        for i in range(len(u)):
            edges.append(
                {
                    "u": int(u[i]),
                    "v": int(v[i]),
                    "weight": float(w[i]),
                    "skel_path": tuple(int(x) for x in paths[i]),
                }
            )
        return nodes_xy, edges

    # Legacy format: NPY (Nx3)
    ep_npy = out_dir / "graph_edges.npy"
    if not ep_npy.exists():
        raise FileNotFoundError(f"Missing {ep_npz} or {ep_npy}")

    edges_arr = np.load(ep_npy)
    edges: List[EdgeTuple] = [(int(a), int(b), float(w)) for (a, b, w) in edges_arr]
    return nodes_xy, edges


# ----------------------------
# Evaluation stage
# ----------------------------
def run_evaluation(
    cfg: Dict[str, Any],
    project_root: Path,
    traversability: np.ndarray,
    nodes_xy: np.ndarray,
    edges: List[Any],
) -> Dict[str, Any]:
    eval_cfg = build_eval_cfg(cfg)
    evaluator = GraphEvaluator(eval_cfg)

    # If edges are routed dicts, convert to (u,v,w) for current evaluator
    edges_for_eval: List[EdgeTuple] = []
    if len(edges) > 0 and isinstance(edges[0], dict):
        edges_for_eval = [(e["u"], e["v"], e["weight"]) for e in edges]
    elif len(edges) > 0 and hasattr(edges[0], "u") and hasattr(edges[0], "v") and hasattr(edges[0], "weight"):
        edges_for_eval = [(e.u, e.v, e.weight) for e in edges]
    else:
        edges_for_eval = edges  # already tuples

    metrics = evaluator.evaluate(traversability, nodes_xy, edges_for_eval)
    evaluator.print_report(metrics)

    out_dir = resolve_under_root(project_root, cfg.get("validation", {}).get("out_dir", "data/validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "evaluation_metrics.json"
    evaluator.save_json(metrics, metrics_path)
    print(f"[INFO] Saved evaluation metrics to: {metrics_path}")
    return metrics


# ----------------------------
# Plotting stage
# ----------------------------
def plot_graph_overlay(
    cfg: Dict[str, Any],
    project_root: Path,
    traversability: np.ndarray,
    nodes_xy: np.ndarray,
    edges: List[Any],
    out_dir: Optional[Path] = None,
) -> None:
    vis_cfg = cfg.get("visualization", {})
    graph_png_path = resolve_under_root(project_root, vis_cfg.get("graph_output_path", "data/traversability_graph_nodes.png"))
    show = bool(vis_cfg.get("show", False))

    # If you have routed edges and a saved skeleton coordinate file, prefer routed plotting
    if len(edges) > 0 and out_dir is not None and (isinstance(edges[0], dict) or hasattr(edges[0], "skel_path")):
        skel_path = out_dir / "skel_coords.npy"
        if skel_path.exists() and hasattr(TraversabilityVisualizer, "plot_graph_overlay_routed"):
            skel_coords = np.load(skel_path)
            viz = TraversabilityVisualizer()
            viz.plot_graph_overlay_routed(
                traversability=traversability,
                nodes_xy=nodes_xy,
                edges=edges,
                skel_coords=skel_coords,
                title="Organic Graph (routed edges)",
                save_path=graph_png_path,
                show=show,
                node_size=20.0,
                edge_width=1.6,
                edge_alpha=0.75,
            )
            return

    # Fallback: legacy straight-edge overlay
    try:
        import networkx as nx
    except Exception:
        print("[WARN] networkx not installed; skipping plot_graph_overlay.")
        return

    H, W = traversability.shape
    G = nx.Graph()
    for i, (x, y) in enumerate(nodes_xy):
        xi = int(np.clip(round(float(x)), 0, W - 1))
        yi = int(np.clip(round(float(y)), 0, H - 1))
        G.add_node(int(i), center=(float(x), float(y)), trav=float(traversability[yi, xi]))

    # edges can be dicts, routed objects, or tuples
    if len(edges) > 0 and isinstance(edges[0], dict):
        iterable = [(e["u"], e["v"], e["weight"]) for e in edges]
    elif len(edges) > 0 and hasattr(edges[0], "u") and hasattr(edges[0], "v") and hasattr(edges[0], "weight"):
        iterable = [(e.u, e.v, e.weight) for e in edges]
    else:
        iterable = edges

    for (i, j, w) in iterable:
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= len(nodes_xy) or jj >= len(nodes_xy):
            continue
        G.add_edge(ii, jj, weight=float(w))

    viz = TraversabilityVisualizer()
    viz.plot_graph_overlay(
        traversability=traversability,
        graph=G,
        title="Organic Graph (straight edges)",
        save_path=graph_png_path,
        show=show,
        edge_alpha=0.75,
        edge_width=0.8,
        node_size=5.5,
    )


# ----------------------------
# Pipeline main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.json", help="Path to config JSON")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    project_root = infer_project_root(config_path)
    cfg = load_config(config_path)

    pipe = cfg.get("pipeline", {})
    run_env = bool(pipe.get("run_environment", True))
    run_graph_flag = bool(pipe.get("run_graph", True))
    run_eval = bool(pipe.get("run_validation", True))

    # --- Environment ---
    if run_env:
        traversability = run_environment(cfg, project_root)
    else:
        map_npy = resolve_under_root(project_root, cfg.get("visualization", {}).get("map_npy_path", "data/traversability_map.npy"))
        if not map_npy.exists():
            raise FileNotFoundError(f"pipeline.run_environment=false but map not found at {map_npy}")
        traversability = np.load(map_npy)
        print(f"[INFO] Loaded traversability map (run_environment=false): {map_npy}")

    # --- Graph ---
    out_dir: Optional[Path] = None
    if run_graph_flag:
        nodes_xy, edges, out_dir = run_graph(cfg, project_root)
        plot_graph_overlay(cfg, project_root, traversability, nodes_xy, edges, out_dir=out_dir)
    else:
        # Load artifacts from resolved output_dir
        gb_cfg = build_graph_build_cfg(cfg)
        out_dir = resolve_under_root(project_root, gb_cfg.output_dir)
        nodes_xy, edges = load_graph_artifacts(out_dir)
        print(f"[INFO] Loaded graph artifacts from: {out_dir}")
        plot_graph_overlay(cfg, project_root, traversability, nodes_xy, edges, out_dir=out_dir)

    # --- Evaluation ---
    if run_eval:
        run_evaluation(cfg, project_root, traversability, nodes_xy, edges)
    else:
        print("[INFO] pipeline.run_validation=false → skipping evaluation.")

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
