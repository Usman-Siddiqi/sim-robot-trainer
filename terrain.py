"""
terrain.py

Author: Usman Siddiqi

Procedural terrain for the biped. The world is a grid of square tiles:

    row    = one terrain type (flat, rough, stairs, slopes, ...)
    column = difficulty level 0 .. num_levels-1

    tile (row, level) is centred at  x = (level + 0.5) * tile_size,  y = (row + 0.5) * tile_size

Every non-flat tile is "radial": the robot starts on a flat platform in the
middle and the obstacle (stairs, slope, bumps, ...) surrounds it as square
rings, so the robot meets it whichever way it walks. "up" tiles are pits the
robot climbs out of, "down" tiles are hills / pyramids it walks down from.

The Terrain class does two jobs that must agree with each other:

  1. build the collision geometry inside the MuJoCo model
       - boxes for anything with vertical faces (stairs, blocks)
       - height fields for smooth shapes (slopes, bumps, waves)
  2. answer "how high is the ground at (x, y)?" for thousands of points at once
     (height scan, rewards, resets) from a pre-computed height raster

Which terrain types exist and how hard they get is configured as a list of
rows, see TerrainCfg.rows (training) and EvalCfg.unseen_terrains (held-out
test terrains the policy never trains on) in config.py.
"""

import mujoco
import numpy as np

RES = 0.025     # resolution of the height lookup raster [m]
HRES = 0.1      # resolution of MuJoCo height fields [m]

_COLORS = {
    "flat": (0.45, 0.48, 0.52), "rough": (0.50, 0.56, 0.44), "stairs_up": (0.48, 0.58, 0.72),
    "stairs_down": (0.78, 0.63, 0.47), "slope_up": (0.52, 0.66, 0.62), "slope_down": (0.72, 0.60, 0.66),
}
_DEFAULT_COLOR = (0.66, 0.62, 0.50)


def _lerp(pair, t):
    """Value of a (level 0, last level) parameter pair at difficulty t in [0, 1]."""
    return pair[0] + (pair[1] - pair[0]) * t


class Tile:
    """Geometry of one tile in local coordinates (origin at the tile centre, ground at z = 0)."""

    def __init__(self, size):
        self.half = size / 2
        self.boxes = []         # (x0, x1, y0, y1, top): solid from z = 0 up to `top`
        self.hfield = None      # (n, n) heights >= 0 on a HRES grid, indexed [ix, iy]

    # ------------------------------------------------------------- helpers
    def grid(self):
        """Coordinates and Chebyshev (square ring) distance of the height field grid points."""
        n = int(round(2 * self.half / HRES)) + 1
        axis = np.linspace(-self.half, self.half, n)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        return x, y, np.maximum(np.abs(x), np.abs(y))

    def add_ring(self, inner, top):
        """Square ring reaching from `inner` to the tile border, as four non-overlapping slabs."""
        h = self.half
        self.boxes += [(inner, h, -h, h, top), (-h, -inner, -h, h, top),
                       (-inner, inner, inner, h, top), (-inner, inner, -h, -inner, top)]

    def add_square(self, size, top):
        self.boxes.append((-size, size, -size, size, top))

    def add_stairs(self, start, runs, rises, direction, base=0.0):
        """
        Square staircase around the centre. Step k begins at distance
        start + runs[0] + ... + runs[k-1] and is rises[k] higher (direction "up")
        or lower ("down") than the previous one.
        """
        runs = [max(RES, round(r / RES) * RES) for r in runs]      # keep edges on the raster grid
        edges = start + np.concatenate([[0.0], np.cumsum(runs)])
        if direction == "up":
            height = base
            for k, rise in enumerate(rises):
                height += rise
                self.add_ring(edges[k], height)
        else:
            height = base + float(np.sum(rises))
            self.add_square(edges[0], height)                       # the top platform
            for k, rise in enumerate(rises[:-1]):
                height -= rise
                self.add_square(edges[k + 1], height)
        return edges[-1]

    # ------------------------------------------------------------ raster
    def rasterize(self):
        """Height of the tile surface on the RES grid (cell centres), indexed [ix, iy]."""
        n = int(round(2 * self.half / RES))
        out = np.zeros((n, n), dtype=np.float32)
        if self.hfield is not None:
            centres = (np.arange(n) + 0.5) * RES / HRES             # in height field index units
            i0 = np.clip(np.floor(centres).astype(int), 0, self.hfield.shape[0] - 2)
            f = (centres - i0).astype(np.float32)
            hf = self.hfield.astype(np.float32)
            a = hf[i0][:, i0] * (1 - f)[None, :] + hf[i0][:, i0 + 1] * f[None, :]
            b = hf[i0 + 1][:, i0] * (1 - f)[None, :] + hf[i0 + 1][:, i0 + 1] * f[None, :]
            out = np.maximum(out, a * (1 - f)[:, None] + b * f[:, None])
        for x0, x1, y0, y1, top in self.boxes:
            i0, i1 = int(round((x0 + self.half) / RES)), int(round((x1 + self.half) / RES))
            j0, j1 = int(round((y0 + self.half) / RES)), int(round((y1 + self.half) / RES))
            out[i0:i1, j0:j1] = np.maximum(out[i0:i1, j0:j1], top)
        return out


# ---------------------------------------------------------------- generators
# Each generator fills a Tile for difficulty t in [0, 1] (level / (num_levels - 1)).

def _gen_flat(tile, row, t, rng, cfg):
    pass


def _gen_rough(tile, row, t, rng, cfg):
    """Random bumps: uniform noise on a 20 cm grid, `height` is the peak-to-peak size."""
    amplitude = _lerp(row["height"], t)
    x, y, dist = tile.grid()
    n = x.shape[0]
    coarse = rng.uniform(0.0, amplitude, ((n + 1) // 2, (n + 1) // 2))
    fine_axis, coarse_axis = np.arange(n), np.arange(coarse.shape[0]) * 2
    rows_interp = np.stack([np.interp(fine_axis, coarse_axis, c) for c in coarse])
    noise = np.stack([np.interp(fine_axis, coarse_axis, c) for c in rows_interp.T]).T
    tile.hfield = noise * np.clip((dist - 0.6) / 0.4, 0.0, 1.0)     # keep the spawn area flat


def _gen_stairs(tile, row, t, rng, cfg):
    """Regular stairs: `rise` grows with the level, the `run` (tread depth) is random per tile."""
    rise = _lerp(row["rise"], t)
    run = rng.uniform(*row["run"])
    run = round(run / RES) * RES
    steps = int(min(row.get("max_steps", 8), np.floor(cfg.obstacle_extent / run + 1e-6)))
    tile.add_stairs(cfg.platform_half, [run] * steps, [rise] * steps, row["direction"])


def _gen_irregular_stairs(tile, row, t, rng, cfg):
    """Every step has its own random rise and run."""
    rise = _lerp(row["rise"], t)
    runs, rises, total = [], [], 0.0
    while True:
        run = round(rng.uniform(*row["run"]) / RES) * RES
        if total + run > cfg.obstacle_extent + 1e-6:
            break
        runs.append(run)
        rises.append(rise * rng.uniform(*row["rise_jitter"]))
        total += run
    tile.add_stairs(cfg.platform_half, runs, rises, row["direction"])


def _gen_landing_stairs(tile, row, t, rng, cfg):
    """Flights of a few steps separated by flat landings, like a staircase in a building."""
    rise, run = _lerp(row["rise"], t), row["run"]
    runs, rises = [], []
    for flight in range(row["flights"]):
        # the last tread of a flight is the landing
        runs += [run] * (row["steps_per_flight"] - 1) + [run if flight == row["flights"] - 1 else row["landing"]]
        rises += [rise] * row["steps_per_flight"]
    tile.add_stairs(cfg.platform_half, runs, rises, row["direction"])


def _gen_slope(tile, row, t, rng, cfg):
    """Square ramp: a bowl ("up") or a hill ("down") with the given slope (rise / run)."""
    slope = _lerp(row["slope"], t)
    x, y, dist = tile.grid()
    ramp = np.clip(dist - cfg.platform_half, 0.0, cfg.obstacle_extent)
    height = slope * (ramp if row["direction"] == "up" else cfg.obstacle_extent - ramp)
    bumps = row.get("roughness", 0.0)
    if bumps:
        height = height + rng.uniform(0.0, bumps, height.shape) * np.clip((dist - 0.8) / 0.2, 0.0, 1.0)
    tile.hfield = height


def _gen_waves(tile, row, t, rng, cfg):
    """Rolling sinusoidal ground, `height` is the peak-to-peak size."""
    amplitude, k = _lerp(row["height"], t), 2 * np.pi / row["wavelength"]
    x, y, dist = tile.grid()
    wave = amplitude * (2.0 - np.cos(k * x) - np.cos(k * y)) / 4.0
    tile.hfield = wave * np.clip((dist - 0.5) / 0.5, 0.0, 1.0)


def _gen_blocks(tile, row, t, rng, cfg):
    """Randomly scattered rectangular blocks of random height (up to `height`)."""
    top = _lerp(row["height"], t)
    reach = cfg.platform_half + cfg.obstacle_extent
    for _ in range(row["count"]):
        w, d = rng.uniform(0.4, 1.2, 2)
        cx, cy = rng.uniform(-reach, reach, 2)
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - d / 2, cy + d / 2
        if max(abs(cx), abs(cy)) - max(w, d) / 2 < 0.7:             # keep the spawn area clear
            continue
        snap = lambda v: round(np.clip(v, -tile.half, tile.half) / RES) * RES
        tile.boxes.append((snap(x0), snap(x1), snap(y0), snap(y1), top * rng.uniform(0.4, 1.0)))


def _gen_ramp_stairs(tile, row, t, rng, cfg):
    """A ramp that leads into a staircase."""
    slope, rise, run = _lerp(row["slope"], t), _lerp(row["rise"], t), row["run"]
    ramp_len = row["ramp_length"]
    x, y, dist = tile.grid()
    tile.hfield = slope * np.clip(dist - cfg.platform_half, 0.0, ramp_len)
    steps = int(np.floor((cfg.obstacle_extent - ramp_len) / run + 1e-6))
    tile.add_stairs(cfg.platform_half + ramp_len, [run] * steps, [rise] * steps, "up", base=slope * ramp_len)


GENERATORS = {
    "flat": _gen_flat, "rough": _gen_rough, "stairs": _gen_stairs, "irregular_stairs": _gen_irregular_stairs,
    "landing_stairs": _gen_landing_stairs, "slope": _gen_slope, "waves": _gen_waves, "blocks": _gen_blocks,
    "ramp_stairs": _gen_ramp_stairs,
}


class Terrain:
    def __init__(self, rows, cfg, seed=0):
        """
        Args:
            rows: list of row dicts, each with "name", "kind" (a key of GENERATORS),
                  "weight" (share of robots) and the parameters of that generator
            cfg: TerrainCfg
            seed: seed for the random parts of the tiles
        """
        self.cfg = cfg
        self.rows = [dict(r) for r in rows]
        if self.rows[0]["kind"] != "flat":
            raise ValueError("the first terrain row must be flat")
        self.num_rows = len(self.rows)
        self.row_names = [r["name"] for r in self.rows]
        self.row_is_flat = np.array([r["kind"] == "flat" for r in self.rows])
        self.num_levels = 1 if self.row_is_flat.all() else cfg.num_levels
        self.tile = float(cfg.tile_size)
        if cfg.platform_half + cfg.obstacle_extent + 0.3 > cfg.success_distance or \
                cfg.success_distance > self.tile / 2 - 0.5:
            raise ValueError("TerrainCfg: tile_size / obstacle_extent / success_distance do not fit together")

        rng = np.random.default_rng(seed)
        n = int(round(self.tile / RES))
        self._raster = np.zeros((self.num_levels * n, self.num_rows * n), dtype=np.float32)
        self._tiles = {}
        for r, row in enumerate(self.rows):
            if row["kind"] == "flat":
                continue
            for level in range(self.num_levels):
                tile = Tile(self.tile)
                t = level / max(self.num_levels - 1, 1)
                GENERATORS[row["kind"]](tile, row, t, rng, cfg)
                self._tiles[(r, level)] = tile
                self._raster[level * n:(level + 1) * n, r * n:(r + 1) * n] = tile.rasterize()
        self._inv_res = 1.0 / RES
        self._max_ix, self._max_iy = self._raster.shape[0] - 1, self._raster.shape[1] - 1

    # ------------------------------------------------------------------ queries
    def tile_center(self, row, level):
        """Centre (x, y) of tiles; row/level can be arrays."""
        return (np.asarray(level) + 0.5) * self.tile, (np.asarray(row) + 0.5) * self.tile

    def height(self, x, y):
        """Terrain height [m] at world points (x, y). Works on arrays of any shape."""
        ix = np.clip(np.floor(np.asarray(x) * self._inv_res), 0, self._max_ix).astype(np.int64)
        iy = np.clip(np.floor(np.asarray(y) * self._inv_res), 0, self._max_iy).astype(np.int64)
        return self._raster[ix, iy].astype(np.float64)

    def distance_from_center(self, x, y, row, level):
        cx, cy = self.tile_center(row, level)
        return np.maximum(np.abs(x - cx), np.abs(y - cy))

    # ----------------------------------------------------------------- geometry
    def add_to_spec(self, spec):
        """Add the ground plane and all tiles to a mujoco.MjSpec."""
        world = spec.worldbody
        world.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
            material="grid", contype=0, conaffinity=1,
        )
        for (r, level), tile in self._tiles.items():
            cx, cy = self.tile_center(r, level)
            base = np.array(_COLORS.get(self.rows[r]["name"], _DEFAULT_COLOR))
            shade = 0.8 + 0.2 * level / max(self.num_levels - 1, 1)
            if tile.hfield is not None and tile.hfield.max() > 1e-4:
                top = float(tile.hfield.max())
                name = f"hfield_{r}_{level}"
                n = tile.hfield.shape[0]
                # MuJoCo stores height fields row by row along y, normalised to [0, 1]
                spec.add_hfield(name=name, nrow=n, ncol=n, size=[tile.half, tile.half, top, 0.05],
                                userdata=(tile.hfield.T / top).ravel().tolist())
                # 2 mm below the ground plane so flat parts do not collide / render twice
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=[cx, cy, -0.002],
                               rgba=[*(base * shade), 1], contype=0, conaffinity=1)
            for i, (x0, x1, y0, y1, top) in enumerate(tile.boxes):
                if top <= 1e-6 or x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6:
                    continue
                tint = shade - 0.05 * (i // 4 % 2 if len(tile.boxes) > 12 else i % 2)
                world.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX, pos=[cx + (x0 + x1) / 2, cy + (y0 + y1) / 2, top / 2],
                    size=[(x1 - x0) / 2, (y1 - y0) / 2, top / 2], rgba=[*(base * tint), 1],
                    contype=0, conaffinity=1,
                )


def build_model(sim_cfg, terrain):
    """Load the robot MJCF, add the terrain and compile to an MjModel."""
    spec = mujoco.MjSpec.from_file(sim_cfg.model_path)
    terrain.add_to_spec(spec)
    model = spec.compile()
    model.opt.timestep = sim_cfg.timestep
    return model
