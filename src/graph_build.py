"""
graph_build.py

Build an "organic" traversability graph on top of an existing traversability map
using:

1) Poisson disk sampling in free space (optionally biased by traversability)
2) CVT (Centroidal Voronoi Tessellation) via Lloyd relaxation (weighted by traversability)
3) Skeletonization of free space to get a medial-axis-like structure
4) Connect CVT sites by routing edges along the skeleton (prevents "grid look")

Outputs:
- nodes: (N, 2) float array in (x, y) map coordinates
- edges: routed edges (RoutedEdge) when skeleton routing is enabled, else (i, j, weight)
- optional debug plots (if TraversabilityVisualizer exists in your repo)

Expected map:
- traversability_map[y, x] in [0..1]
- walls/rocks == 0
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from typing import Sequence

import numpy as np

try:
    import networkx as nx
except Exception as e:
    raise ImportError("graph_build.py requires networkx. Please install it.") from e

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError("graph_build.py requires scipy (cKDTree). Please install it.") from e


# ----------------------------
# Optional skeleton dependency
# ----------------------------
_SKIMAGE_AVAILABLE = False
try:
    from skimage.morphology import skeletonize
    _SKIMAGE_AVAILABLE = True
except Exception:
    _SKIMAGE_AVAILABLE = False


# ----------------------------
# Optional project visualizer
# ----------------------------
_VIS_AVAILABLE = False
try:
    from visualization import TraversabilityVisualizer
    _VIS_AVAILABLE = True
except Exception:
    _VIS_AVAILABLE = False


@dataclass(frozen=True)
class RoutedEdge:
    u: int
    v: int
    weight: float
    skel_path: Tuple[int, ...]   # skeleton node ids along the route (inclusive)


@dataclass
class GraphBuildConfig:
    map_path: str = "data/traversability_map.npy"
    output_dir: str = "data"

    # Poisson disk sampling
    use_poisson_sampling: bool = True
    poisson_radius: float = 6.0     # in cells/pixels
    poisson_k: int = 30             # candidates per active point
    poisson_bias: float = 0.0       # 0 = uniform, >0 biases toward higher traversability

    # CVT (Lloyd)
    use_cvt: bool = True
    cvt_iters: int = 15
    cvt_sample_stride: int = 2      # downsample grid for speed in CVT assignment
    cvt_weight_power: float = 1.0   # weight = traversability**power
    cvt_keep_in_free: bool = True

    # Skeleton + edges
    use_skeleton_routing: bool = True
    skeleton_prune_spurs: bool = True
    spur_max_length: int = 10       # pixels
    max_snap_dist: float = 10.0     # max distance from CVT site to skeleton to allow snapping
    max_edge_length: float = 200.0  # reject very long connections
    max_degree: int = 6             # cap degree per node after building edges (keeps graph tidy)
    seed_stride: int = 4            # used if poisson is disabled
    knn_k: int = 8                  # used if skeleton routing is disabled

    # Debug images
    save_debug_images: bool = True


# ----------------------------
# Utilities
# ----------------------------
def _load_config(config_path: Optional[str]) -> GraphBuildConfig:
    cfg = GraphBuildConfig()
    if not config_path:
        return cfg
    with open(config_path, "r") as f:
        data = json.load(f)
    # allow nesting under e.g. "graph_build"
    block = data.get("graph_build", data)
    for k, v in block.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _free_mask(trav: np.ndarray) -> np.ndarray:
    return trav > 0.0


def _grid_samples_from_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    h, w = mask.shape
    samples = []
    for y in range(0, h, stride):
        row = mask[y]
        for x in range(0, w, stride):
            if row[x]:
                samples.append((float(x), float(y)))
    return np.asarray(samples, dtype=float)


def connect_sites_knn(
    sites: np.ndarray,
    max_edge_length: float,
    max_degree: int,
    k: int,
) -> List[Tuple[int, int, float]]:
    if len(sites) == 0:
        return []
    k = max(1, int(k))
    tree = cKDTree(sites)
    n = len(sites)
    query_k = min(k + 1, n)
    if query_k == 1:
        return []
    dists, idxs = tree.query(sites, k=query_k)

    degrees = [0] * n
    seen: set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int, float]] = []

    for i in range(n):
        di = dists[i]
        ii = idxs[i]
        for dist, j in zip(di[1:], ii[1:]):
            jj = int(j)
            if jj == i:
                continue
            if dist <= 0 or dist > max_edge_length:
                continue
            a, b = (i, jj) if i < jj else (jj, i)
            if (a, b) in seen:
                continue
            if degrees[a] >= max_degree or degrees[b] >= max_degree:
                continue
            seen.add((a, b))
            degrees[a] += 1
            degrees[b] += 1
            edges.append((a, b, float(dist)))

    return edges


def _clip_point(p: np.ndarray, w: int, h: int) -> np.ndarray:
    p[0] = np.clip(p[0], 0, w - 1)
    p[1] = np.clip(p[1], 0, h - 1)
    return p


# ----------------------------
# 1) Poisson disk sampling (Bridson)
# ----------------------------
def poisson_disk_sampling_masked(
    mask: np.ndarray,
    radius: float,
    k: int = 30,
    rng: Optional[np.random.Generator] = None,
    trav: Optional[np.ndarray] = None,
    bias: float = 0.0,
) -> np.ndarray:
    """
    Poisson disk sampling over pixels where mask==True.

    If trav and bias>0: acceptance probability ~ (trav[y,x] ** bias)
    which biases samples toward higher traversability while keeping Poisson spacing.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = mask.shape
    if radius <= 0:
        raise ValueError("radius must be > 0")

    # grid cell size for acceleration
    cell_size = radius / math.sqrt(2)
    grid_w = int(math.ceil(w / cell_size))
    grid_h = int(math.ceil(h / cell_size))
    grid = -np.ones((grid_h, grid_w), dtype=int)

    def grid_coords(pt: np.ndarray) -> Tuple[int, int]:
        return int(pt[1] / cell_size), int(pt[0] / cell_size)

    def in_neighborhood(pt: np.ndarray, pts: List[np.ndarray]) -> bool:
        gy, gx = grid_coords(pt)
        y0 = max(gy - 2, 0)
        y1 = min(gy + 3, grid_h)
        x0 = max(gx - 2, 0)
        x1 = min(gx + 3, grid_w)
        r2 = radius * radius
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                idx = grid[yy, xx]
                if idx == -1:
                    continue
                q = pts[idx]
                if (pt[0] - q[0]) ** 2 + (pt[1] - q[1]) ** 2 < r2:
                    return True
        return False

    # pick initial point in free space
    free_idx = np.argwhere(mask)
    if free_idx.size == 0:
        return np.zeros((0, 2), dtype=float)

    # Try a few times to pick a good seed if biased
    for _ in range(50):
        yx = free_idx[rng.integers(0, len(free_idx))]
        x0, y0 = float(yx[1]), float(yx[0])
        if trav is not None and bias > 0:
            p = float(trav[int(y0), int(x0)]) ** bias
            if rng.random() > p:
                continue
        seed = np.array([x0, y0], dtype=float)
        break
    else:
        yx = free_idx[rng.integers(0, len(free_idx))]
        seed = np.array([float(yx[1]), float(yx[0])], dtype=float)

    points: List[np.ndarray] = [seed]
    active: List[int] = [0]

    gy, gx = grid_coords(seed)
    grid[gy, gx] = 0

    while active:
        a_idx = active[rng.integers(0, len(active))]
        base = points[a_idx]
        found = False

        for _ in range(k):
            ang = rng.random() * 2 * math.pi
            rad = radius * (1 + rng.random())
            cand = np.array([base[0] + rad * math.cos(ang), base[1] + rad * math.sin(ang)], dtype=float)

            if cand[0] < 0 or cand[0] >= w or cand[1] < 0 or cand[1] >= h:
                continue

            cx, cy = int(cand[0]), int(cand[1])
            if not mask[cy, cx]:
                continue

            if trav is not None and bias > 0:
                acc = float(trav[cy, cx]) ** bias
                if rng.random() > acc:
                    continue

            if in_neighborhood(cand, points):
                continue

            points.append(cand)
            idx = len(points) - 1
            active.append(idx)
            gyy, gxx = grid_coords(cand)
            grid[gyy, gxx] = idx
            found = True
            break

        if not found:
            active.remove(a_idx)

    return np.asarray(points, dtype=float)


# ----------------------------
# 2) CVT via Lloyd relaxation (weighted)
# ----------------------------
def cvt_lloyd_weighted(
    sites: np.ndarray,
    trav: np.ndarray,
    iters: int,
    sample_stride: int = 2,
    weight_power: float = 1.0,
    keep_in_free: bool = True,
) -> np.ndarray:
    """
    Approximate weighted CVT on a grid:
      - Assign sampled grid points to nearest site (KDTree)
      - Move each site to the weighted centroid of its assigned samples
    """
    if len(sites) == 0:
        return sites

    h, w = trav.shape
    mask = trav > 0.0
    sites = sites.copy()

    ys = np.arange(0, h, sample_stride)
    xs = np.arange(0, w, sample_stride)
    grid_pts = np.array([(x, y) for y in ys for x in xs], dtype=float)

    # Keep only free points
    free_flags = mask[grid_pts[:, 1].astype(int), grid_pts[:, 0].astype(int)]
    grid_pts = grid_pts[free_flags]
    if len(grid_pts) == 0:
        return sites

    # weights from traversability
    tvals = trav[grid_pts[:, 1].astype(int), grid_pts[:, 0].astype(int)]
    wts = np.power(np.clip(tvals, 0.0, 1.0), weight_power).astype(float)
    wts = np.maximum(wts, 1e-6)

    for _ in range(iters):
        tree = cKDTree(sites)
        _, nn = tree.query(grid_pts, k=1)

        new_sites = sites.copy()
        for i in range(len(sites)):
            sel = (nn == i)
            if not np.any(sel):
                continue
            pts_i = grid_pts[sel]
            w_i = wts[sel][:, None]
            centroid = (pts_i * w_i).sum(axis=0) / (w_i.sum(axis=0) + 1e-12)
            centroid = _clip_point(centroid, w, h)

            if keep_in_free:
                cx, cy = int(round(centroid[0])), int(round(centroid[1]))
                if not mask[cy, cx]:
                    # snap to nearest free pixel
                    # (cheap approach: local search expanding square)
                    snapped = _snap_to_nearest_free(mask, centroid, max_r=15)
                    if snapped is not None:
                        centroid = snapped

            new_sites[i] = centroid

        sites = new_sites

    return sites


def _snap_to_nearest_free(mask: np.ndarray, p: np.ndarray, max_r: int = 15) -> Optional[np.ndarray]:
    h, w = mask.shape
    x0, y0 = int(round(p[0])), int(round(p[1]))
    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return None
    if mask[y0, x0]:
        return np.array([float(x0), float(y0)], dtype=float)

    for r in range(1, max_r + 1):
        x_min, x_max = max(0, x0 - r), min(w - 1, x0 + r)
        y_min, y_max = max(0, y0 - r), min(h - 1, y0 + r)
        # perimeter scan
        coords = []
        for x in range(x_min, x_max + 1):
            coords.append((x, y_min))
            coords.append((x, y_max))
        for y in range(y_min + 1, y_max):
            coords.append((x_min, y))
            coords.append((x_max, y))
        for (x, y) in coords:
            if mask[y, x]:
                return np.array([float(x), float(y)], dtype=float)
    return None


# ----------------------------
# 3) Skeletonization + skeleton graph
# ----------------------------
def skeletonize_free_space(mask: np.ndarray) -> np.ndarray:
    """
    Returns a boolean skeleton image.
    Requires scikit-image. If unavailable, we fall back to a *very* rough method:
    boundary erosion loops until thin (not as good as true skeleton).
    """
    if _SKIMAGE_AVAILABLE:
        # skeletonize expects foreground True
        skel = skeletonize(mask.astype(bool))
        return skel.astype(bool)

    # Fallback: iterative erosion with simple thinning heuristic (not great, but works)
    # NOTE: This is a last resort; install scikit-image for proper skeletons.
    skel = mask.copy().astype(bool)
    changed = True
    while changed:
        changed = False
        to_remove = []
        ys, xs = np.where(skel)
        for y, x in zip(ys, xs):
            # Keep endpoints/junctions by neighbor count
            n = _count_8_neighbors(skel, x, y)
            if n <= 1:
                continue
            # remove boundary-ish pixels first
            if _count_8_neighbors(mask, x, y) < 8:
                to_remove.append((y, x))
        if to_remove:
            for y, x in to_remove:
                skel[y, x] = False
            changed = True
    return skel


def _count_8_neighbors(img: np.ndarray, x: int, y: int) -> int:
    h, w = img.shape
    c = 0
    for yy in range(max(0, y - 1), min(h, y + 2)):
        for xx in range(max(0, x - 1), min(w, x + 2)):
            if yy == y and xx == x:
                continue
            c += 1 if img[yy, xx] else 0
    return c


def build_skeleton_graph(skel: np.ndarray) -> Tuple[nx.Graph, Dict[Tuple[int, int], int]]:
    """
    Builds an undirected graph from skeleton pixels (8-neighborhood).
    Nodes are integer pixel coords encoded as IDs.
    Returns graph + mapping from (x,y) -> node_id
    """
    h, w = skel.shape
    G = nx.Graph()
    coord_to_id: Dict[Tuple[int, int], int] = {}

    ys, xs = np.where(skel)
    for (y, x) in zip(ys.tolist(), xs.tolist()):
        nid = len(coord_to_id)
        coord_to_id[(x, y)] = nid
        G.add_node(nid, x=x, y=y)

    # Add edges between 8-neighbors
    for (x, y), nid in coord_to_id.items():
        for yy in range(max(0, y - 1), min(h, y + 2)):
            for xx in range(max(0, x - 1), min(w, x + 2)):
                if xx == x and yy == y:
                    continue
                if not skel[yy, xx]:
                    continue
                nid2 = coord_to_id.get((xx, yy))
                if nid2 is None:
                    continue
                # weight = Euclidean step
                w_step = math.hypot(xx - x, yy - y)
                if nid2 != nid:
                    G.add_edge(nid, nid2, weight=w_step)

    return G, coord_to_id


def prune_skeleton_spurs(G: nx.Graph, max_len: int = 10) -> nx.Graph:
    """
    Remove short dangling spurs (chains from degree-1 endpoints).
    """
    G = G.copy()
    while True:
        endpoints = [n for n in G.nodes if G.degree[n] == 1]
        removed_any = False
        for ep in endpoints:
            # walk until junction or length exceeds max_len
            path = [ep]
            cur = ep
            length = 0.0
            prev = None
            while True:
                nbrs = list(G.neighbors(cur))
                if prev is not None:
                    nbrs = [v for v in nbrs if v != prev]
                if len(nbrs) == 0:
                    break
                nxt = nbrs[0]
                length += G[cur][nxt].get("weight", 1.0)
                path.append(nxt)
                prev, cur = cur, nxt
                if G.degree[cur] != 2:
                    break
                if length > max_len:
                    break

            if length <= max_len and len(path) >= 2:
                # remove all nodes in the spur except the last (junction)
                for n in path[:-1]:
                    if n in G:
                        G.remove_node(n)
                removed_any = True

        if not removed_any:
            break
    return G


# ----------------------------
# 4) Snap CVT sites to skeleton and connect via shortest paths
# ----------------------------
def snap_sites_to_skeleton(
    sites: np.ndarray,
    skel_coords: np.ndarray,
    max_snap_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Snap each site to nearest skeleton pixel.
    Returns:
      - snapped_xy (N,2) float
      - snapped_index (N,) int index into skel_coords, or -1 if too far
    """
    if len(sites) == 0 or len(skel_coords) == 0:
        return sites.copy(), -np.ones((len(sites),), dtype=int)

    tree = cKDTree(skel_coords)
    d, idx = tree.query(sites, k=1)
    snapped = skel_coords[idx].astype(float)
    ok = d <= max_snap_dist
    snapped_idx = np.where(ok, idx, -1)
    snapped_xy = np.where(ok[:, None], snapped, sites)
    return snapped_xy, snapped_idx.astype(int)


def connect_sites_via_skeleton(
    sites: np.ndarray,
    snapped_idx: np.ndarray,
    skel_graph: nx.Graph,
    skel_coords: np.ndarray,
    max_edge_length: float,
    max_degree: int,
) -> List[RoutedEdge]:
    """
    Build edges between CVT sites by shortest paths on the skeleton graph.

    Returns RoutedEdge(u, v, weight, skel_path) where:
      - weight is the skeleton shortest-path length
      - skel_path is the list/tuple of skeleton node IDs along the path
    """
    N = len(sites)
    if N == 0:
        return []

    site_tree = cKDTree(sites)

    # Propose local candidate edges (increase a bit for better connectivity)
    # You can tune this; 20 is usually safe for N~40-200
    k = min(20, max(4, int(max_degree * 4)))
    _, neigh = site_tree.query(sites, k=min(k, N))

    # Store best candidate per (a,b)
    candidates: Dict[Tuple[int, int], RoutedEdge] = {}

    for i in range(N):
        for j in neigh[i]:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a == b:
                continue

            si = int(snapped_idx[a])
            sj = int(snapped_idx[b])
            if si < 0 or sj < 0:
                continue
            if (si not in skel_graph) or (sj not in skel_graph):
                continue

            try:
                path = nx.shortest_path(skel_graph, si, sj, weight="weight")
                dist = nx.path_weight(skel_graph, path, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if dist <= 0 or dist > max_edge_length:
                continue

            edge = RoutedEdge(u=a, v=b, weight=float(dist), skel_path=tuple(int(p) for p in path))

            # Keep the shortest if duplicate candidate appears
            prev = candidates.get((a, b))
            if prev is None or edge.weight < prev.weight:
                candidates[(a, b)] = edge

    # Enforce max_degree by greedy pruning on each node (keep smallest weights)
    adj: Dict[int, List[RoutedEdge]] = {i: [] for i in range(N)}
    for e in candidates.values():
        adj[e.u].append(e)
        adj[e.v].append(e)

    kept_keys = set(candidates.keys())

    for i in range(N):
        nbrs = sorted(adj[i], key=lambda e: e.weight)
        for e in nbrs[max_degree:]:
            key = (e.u, e.v)
            if key in kept_keys:
                kept_keys.remove(key)

    out = [candidates[k] for k in kept_keys]
    return out



# ----------------------------
# Main build function
# ----------------------------
def build_graph_from_traversability(cfg: GraphBuildConfig) -> Tuple[np.ndarray, List[Any]]:
    map_path = Path(cfg.map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Traversability map not found: {map_path}")

    trav = np.load(map_path)
    if trav.ndim != 2:
        raise ValueError(f"Expected 2D traversability map, got shape={trav.shape}")

    mask = _free_mask(trav)
    h, w = trav.shape

    rng = np.random.default_rng()

    # 1) Poisson samples (or grid fallback)
    if cfg.use_poisson_sampling:
        samples = poisson_disk_sampling_masked(
            mask=mask,
            radius=float(cfg.poisson_radius),
            k=int(cfg.poisson_k),
            rng=rng,
            trav=trav,
            bias=float(cfg.poisson_bias),
        )
    else:
        samples = _grid_samples_from_mask(mask, stride=int(cfg.seed_stride))

    # 2) CVT relaxation
    if cfg.use_cvt and len(samples) > 0:
        sites = cvt_lloyd_weighted(
            sites=samples,
            trav=trav,
            iters=int(cfg.cvt_iters),
            sample_stride=int(cfg.cvt_sample_stride),
            weight_power=float(cfg.cvt_weight_power),
            keep_in_free=bool(cfg.cvt_keep_in_free),
        )
    else:
        sites = np.asarray(samples, dtype=float)

    # 3-5) Skeleton routing or straight kNN edges
    skel = None
    if cfg.use_skeleton_routing:
        skel = skeletonize_free_space(mask)

        skel_graph, coord_to_id = build_skeleton_graph(skel)
        if cfg.skeleton_prune_spurs:
            skel_graph = prune_skeleton_spurs(skel_graph, max_len=int(cfg.spur_max_length))

        # Rebuild skel_coords aligned with node ids:
        # node ids are ints, we assume they are 0..n-1, but pruning can create holes.
        # We'll relabel for stable shortest-path usage.
        skel_graph = nx.convert_node_labels_to_integers(skel_graph, first_label=0, ordering="default")
        skel_coords = np.zeros((skel_graph.number_of_nodes(), 2), dtype=float)
        for nid, data in skel_graph.nodes(data=True):
            skel_coords[nid] = [float(data["x"]), float(data["y"])]

        # 4) Snap sites to skeleton
        snapped_xy, snapped_idx = snap_sites_to_skeleton(
            sites=sites,
            skel_coords=skel_coords,
            max_snap_dist=float(cfg.max_snap_dist),
        )

        # 5) Connect via skeleton shortest paths
        edges = connect_sites_via_skeleton(
            sites=snapped_xy,
            snapped_idx=snapped_idx,
            skel_graph=skel_graph,
            skel_coords=skel_coords,
            max_edge_length=float(cfg.max_edge_length),
            max_degree=int(cfg.max_degree),
        )
        nodes_xy = snapped_xy
    else:
        skel_graph = nx.Graph()
        skel_coords = np.zeros((0, 2), dtype=float)
        nodes_xy = np.asarray(sites, dtype=float)
        edges = connect_sites_knn(
            sites=nodes_xy,
            max_edge_length=float(cfg.max_edge_length),
            max_degree=int(cfg.max_degree),
            k=int(cfg.knn_k),
        )

    # Save outputs
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "graph_nodes.npy", nodes_xy.astype(np.float32))

    if cfg.use_skeleton_routing:
        # Save routed edges as NPZ (ragged paths)
        u = np.array([e.u for e in edges], dtype=np.int32)
        v = np.array([e.v for e in edges], dtype=np.int32)
        w = np.array([e.weight for e in edges], dtype=np.float32)
        paths = np.array([np.array(e.skel_path, dtype=np.int32) for e in edges], dtype=object)

        np.savez(out_dir / "graph_edges.npz", u=u, v=v, w=w, paths=paths)
    else:
        np.save(out_dir / "graph_edges.npy", np.asarray(edges, dtype=np.float32))


    if cfg.save_debug_images and _VIS_AVAILABLE:
        vis = TraversabilityVisualizer()
        # Provide a few helpful debug plots if your visualizer supports it
        # (these method names are guesses — adapt to your actual visualizer API)
        try:
            vis.plot_traversability(trav, filename="trav_map.png")
        except Exception:
            pass
        try:
            vis.plot_points(trav, points=samples, filename="poisson_samples.png")
        except Exception:
            pass
        try:
            vis.plot_points(trav, points=nodes_xy, filename="cvt_sites.png")
        except Exception:
            pass
        try:
            if skel is not None:
                vis.plot_skeleton(trav, skel=skel, filename="skeleton.png")
        except Exception:
            pass
        try:
            vis.plot_graph(trav, nodes=nodes_xy, edges=edges, filename="graph_organic.png")
        except Exception:
            pass

    print(f"[INFO] Loaded traversability map: {map_path} shape={trav.shape}")
    print(f"[INFO] Poisson samples: {len(samples)}")
    print(f"[INFO] CVT sites: {len(sites)}")
    print(f"[INFO] Skeleton nodes: {skel_graph.number_of_nodes()} edges: {skel_graph.number_of_edges()}")
    print(f"[INFO] Graph edges: {len(edges)}")
    print(f"[INFO] Saved nodes to: {out_dir / 'graph_nodes.npy'}")
    if cfg.use_skeleton_routing:
        print(f"[INFO] Saved edges to: {out_dir / 'graph_edges.npz'}")
    else:
        print(f"[INFO] Saved edges to: {out_dir / 'graph_edges.npy'}")

    return nodes_xy, edges


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to config JSON (optional)")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    # Sensible default if user didn’t install scikit-image
    if not _SKIMAGE_AVAILABLE:
        print("[WARN] scikit-image not found. Skeleton fallback is crude. Install scikit-image for best results.")

    build_graph_from_traversability(cfg)


if __name__ == "__main__":
    main()
