"""
graph_trav.py

Environment generation for a 2D traversability map, configured via
config/config.json.

Supports:
- Perlin-based open terrain ("perlin")
- Cave-like underground map using cellular automata ("cave")

No graph / pathfinding logic here - just map generation + plotting.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

# Headless-friendly backend (Docker, no X server)
matplotlib.use("Agg")

from perlin_noise import PerlinNoise

from visualization import TraversabilityVisualizer


class EnvironmentGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        map_generator: str,
        perlin_scale: float,
        perlin_octaves: int,
        perlin_seed: int,
        cave_fill_prob: float,
        cave_birth_limit: int,
        cave_death_limit: int,
        cave_steps: int,
        cave_min_traversability: float,
    ):
        """
        :param width:  number of cells in X
        :param height: number of cells in Y
        :param map_generator: "perlin" or "cave"
        :param perlin_scale: scale factor for Perlin coordinates
        :param perlin_octaves: number of octaves for Perlin noise
        :param perlin_seed: random seed for Perlin noise
        :param cave_fill_prob: initial probability of a cell being rock (1)
        :param cave_birth_limit: CA rule: empty cell -> rock if neighbors_rock > birth_limit
        :param cave_death_limit: CA rule: rock cell -> empty if neighbors_rock < death_limit
        :param cave_steps: number of CA steps to run
        :param cave_min_traversability: minimum traversability value inside tunnels (0–1)
        """
        self.width = width
        self.height = height
        self.map_generator = map_generator.lower()

        self.perlin_scale = perlin_scale
        self.perlin_octaves = perlin_octaves
        self.perlin_seed = perlin_seed

        self.cave_fill_prob = cave_fill_prob
        self.cave_birth_limit = cave_birth_limit
        self.cave_death_limit = cave_death_limit
        self.cave_steps = cave_steps
        self.cave_min_traversability = cave_min_traversability

        self.traversability: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 1) Public API
    # ------------------------------------------------------------------ #
    def generate_traversability_map(self) -> np.ndarray:
        """
        Generate the traversability map according to the selected generator.

        Returns:
            traversability map as a (height, width) numpy array in [0, 1].
        """
        if self.map_generator == "perlin":
            trav = self._generate_perlin_map()
        elif self.map_generator == "cave":
            trav = self._generate_cave_map()
        else:
            raise ValueError("map_generator must be 'perlin' or 'cave'")

        self.traversability = trav
        return trav


    # ------------------------------------------------------------------ #
    # 2) Perlin-based map
    # ------------------------------------------------------------------ #
    def _generate_perlin_map(self) -> np.ndarray:
        """Perlin-noise-based traversability map in [0, 1]."""
        noise = PerlinNoise(octaves=self.perlin_octaves, seed=self.perlin_seed)
        trav = np.zeros((self.height, self.width), dtype=float)

        for y in range(self.height):
            for x in range(self.width):
                n = noise([x / self.perlin_scale, y / self.perlin_scale])  # [-1, 1]
                trav[y, x] = (n + 1.0) / 2.0  # -> [0, 1]

        return trav

    # ------------------------------------------------------------------ #
    # 3) Cave-based map (binary cave + gradient traversability)
    # ------------------------------------------------------------------ #
    def _generate_cave_map(self) -> np.ndarray:
        """
        Generate a cave-like underground map using cellular automata.

        Internal representation during CA:
            1 = rock (wall)
            0 = empty space (tunnel)

        Final traversability:
            rock -> 0.0
            tunnels -> gradient in [min_trav, 1.0]
        """
        rng = np.random.default_rng(self.perlin_seed)

        # 1 = rock, 0 = empty (initial random noise)
        cave = (rng.random((self.height, self.width)) < self.cave_fill_prob).astype(
            np.int8
        )

        # Run CA smoothing
        for _ in range(self.cave_steps):
            cave = self._cave_step(cave)

        # Borders as rock (we're inside a mine / cave volume)
        cave[0, :] = 1
        cave[-1, :] = 1
        cave[:, 0] = 1
        cave[:, -1] = 1

        # Keep only the largest connected empty region (one big cave)
        cave = self._keep_largest_region(cave)

        # Re-enforce borders as rock (just in case)
        cave[0, :] = 1
        cave[-1, :] = 1
        cave[:, 0] = 1
        cave[:, -1] = 1

        # --- Convert to traversability with gradient inside cave ---

        # mask: 1 in free space, 0 in rock
        mask = 1.0 - cave.astype(float)  # free = 1, rock = 0

        # Perlin noise field for the interior
        noise = PerlinNoise(octaves=self.perlin_octaves, seed=self.perlin_seed + 1)
        noise_field = np.zeros_like(mask)

        for y in range(self.height):
            for x in range(self.width):
                n = noise([x / self.perlin_scale, y / self.perlin_scale])  # [-1, 1]
                noise_field[y, x] = (n + 1.0) / 2.0  # [0, 1]

        # Minimum traversability inside the cave (from config)
        min_trav = float(self.cave_min_traversability)
        min_trav = np.clip(min_trav, 0.0, 1.0)

        # Inside cave: min_trav .. 1.0 ; in rock: 0.0
        trav = np.where(
            mask > 0.0,
            min_trav + (1.0 - min_trav) * noise_field,
            0.0,
        )

        return trav

    def _cave_step(self, cave: np.ndarray) -> np.ndarray:
        """
        One step of the cellular automaton for cave generation.

        For each cell, count the number of rock neighbors (8-connected).
        Apply rules:
            If cell is rock:
                stays rock if neighbor_rock >= death_limit, else becomes empty
            If cell is empty:
                becomes rock if neighbor_rock > birth_limit, else stays empty
        """
        H, W = cave.shape
        new_cave = np.zeros_like(cave)

        for y in range(H):
            for x in range(W):
                rock_neighbors = self._count_rock_neighbors(cave, x, y)

                if cave[y, x] == 1:
                    # rock cell
                    if rock_neighbors >= self.cave_death_limit:
                        new_cave[y, x] = 1
                    else:
                        new_cave[y, x] = 0
                else:
                    # empty cell
                    if rock_neighbors > self.cave_birth_limit:
                        new_cave[y, x] = 1
                    else:
                        new_cave[y, x] = 0

        return new_cave

    @staticmethod
    def _count_rock_neighbors(cave: np.ndarray, x: int, y: int) -> int:
        """Count rock cells in the 8-neighborhood around (x, y)."""
        H, W = cave.shape
        count = 0

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < W and 0 <= ny_ < H:
                    count += cave[ny_, nx_]
                else:
                    # Treat out-of-bounds as rock to help close caves at edges
                    count += 1

        return count

    def _keep_largest_region(self, cave: np.ndarray) -> np.ndarray:
        """
        Find all connected empty regions (0 = empty) using flood fill,
        keep ONLY the largest one, fill the rest with rock (1).

        This guarantees a single connected cave system.
        """
        H, W = cave.shape
        visited = np.zeros_like(cave, dtype=bool)
        regions: list[list[tuple[int, int]]] = []

        def flood_fill(sx: int, sy: int) -> list[tuple[int, int]]:
            stack = [(sx, sy)]
            region = [(sx, sy)]
            visited[sy, sx] = True

            while stack:
                x, y = stack.pop()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < W and 0 <= ny_ < H:
                        if not visited[ny_, nx_] and cave[ny_, nx_] == 0:
                            visited[ny_, nx_] = True
                            region.append((nx_, ny_))
                            stack.append((nx_, ny_))
            return region

        # Find all empty connected regions
        for y in range(H):
            for x in range(W):
                if cave[y, x] == 0 and not visited[y, x]:
                    regions.append(flood_fill(x, y))

        if not regions:
            # degenerate case: everything is rock
            return cave

        # Keep only the largest empty region
        largest = max(regions, key=len)
        new_cave = np.ones_like(cave, dtype=np.int8)  # start as all rock
        for (x, y) in largest:
            new_cave[y, x] = 0  # open tunnel

        return new_cave


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
    cave_cfg = cfg.get("cave", {})
    vis_cfg = cfg.get("visualization", {})

    width = int(map_cfg.get("width", 100))
    height = int(map_cfg.get("height", 100))
    map_generator = str(map_cfg.get("generator", "cave"))
    use_existing = bool(map_cfg.get("use_existing", True))

    perlin_scale = float(perlin_cfg.get("scale", 30.0))
    perlin_octaves = int(perlin_cfg.get("octaves", 3))
    perlin_seed = int(perlin_cfg.get("seed", 0))

    cave_fill_prob = float(cave_cfg.get("fill_probability", 0.45))
    cave_birth_limit = int(cave_cfg.get("birth_limit", 4))
    cave_death_limit = int(cave_cfg.get("death_limit", 3))
    cave_steps = int(cave_cfg.get("steps", 5))
    cave_min_trav = float(cave_cfg.get("min_traversability", 0.3))  # keep default here

    map_npy_path_str = vis_cfg.get("map_npy_path", "data/traversability_map.npy")
    env_output_path_str = vis_cfg.get("env_output_path", "data/traversability_map.png")
    show_flag = bool(vis_cfg.get("show", False))

    project_root = Path(__file__).resolve().parents[1]
    map_npy_path = project_root / map_npy_path_str
    env_output_path = project_root / env_output_path_str

    trav = None

    # --- Load or generate ---
    if use_existing and map_npy_path.exists():
        print(f"[INFO] Using existing traversability map: {map_npy_path}")
        trav = np.load(map_npy_path)
    else:
        if use_existing:
            print(f"[WARN] map_npy_path not found, generating new map: {map_npy_path}")
        else:
            print(f"[INFO] Forced regeneration (use_existing=false).")

        env = EnvironmentGenerator(
            width=width,
            height=height,
            map_generator=map_generator,
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

    # --- Always plot from trav (loaded or generated) ---
    print(f"[INFO] Saving environment PNG to: {env_output_path.resolve()}")
    viz = TraversabilityVisualizer()
    viz.plot_traversability_map(
        traversability=trav,
        title=f"Traversability Map ({map_generator})",
        save_path=env_output_path,
        show=show_flag,
    )


if __name__ == "__main__":
    main()
