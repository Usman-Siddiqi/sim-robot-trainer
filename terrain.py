"""
terrain.py

Author: Usman Siddiqi

Procedural hiking trails. A trail is a raised, winding path made of square rock
columns (HikeCfg.cell wide), running from one end of a tile (x = -6 m) to the
other (x = +6 m). Next to the trail the ground is far below, so stepping off it -
or into the gap between two stepping stones - means falling.

    row    = one kind of trail ("biome": rocky, climb, stones, ridge, ...)
    level  = difficulty 0 .. num_levels-1: the trail gets narrower and swings
             further left / right, steps get taller, gaps wider

A trail is assembled from sections, each 2-3 m long:

    flat        plain trail                      rough     bumpy paving
    steps       irregular rock steps up / down   steps_up / steps_down   a climb / a descent
    stones      stepping stones over a gap       ridge     a narrow ridge
    boulders    boulders next to the trail       (and for testing only: logs, terraces, rubble, stairs)

The Terrain class does two jobs that must agree with each other:

  1. build the collision geometry: one MuJoCo model per tile, in which the
     columns are merged into as few boxes as possible
  2. answer "how high is the ground at (x, y)?" for thousands of points at once
     (height scan, rewards, resets) from a pre-computed height raster

Every tile also knows its path (centre line, elevation, width), which the
environment uses to steer the robot and to notice when it left the trail.
"""

import mujoco
import numpy as np

from config import HikeCfg

RES = 0.025         # resolution of the height lookup raster [m]
QUANT = 0.005       # column heights are multiples of this [m]

_COLORS = {
    "flat": (0.45, 0.48, 0.52), "rocky": (0.62, 0.58, 0.50), "climb": (0.56, 0.60, 0.68),
    "stones": (0.52, 0.64, 0.60), "ridge": (0.70, 0.60, 0.52), "trail": (0.64, 0.62, 0.54),
}
_DEFAULT_COLOR = (0.66, 0.58, 0.62)


def _lerp(pair, t):
    """Value of a (level 0, last level) parameter pair at difficulty t in [0, 1]."""
    return pair[0] + (pair[1] - pair[0]) * t


class Tile:
    """One trail. Arrays are indexed [ix, iy] (columns along the trail, cells across)."""

    def __init__(self, cfg):
        self.nx = int(round(cfg.tile_length / cfg.cell))
        self.ny = int(round(cfg.tile_width / cfg.cell))
        self.x = -cfg.tile_length / 2 + (np.arange(self.nx) + 0.5) * cfg.cell
        self.y = -cfg.tile_width / 2 + (np.arange(self.ny) + 0.5) * cfg.cell
        self.height = np.zeros((self.nx, self.ny))      # top of every rock column (0 = no rock)
        self.path_y = np.zeros(self.nx)                 # centre of the trail at every x
        self.path_elev = np.zeros(self.nx)              # nominal height of the trail
        self.path_hw = np.full(self.nx, 1e3)            # half width of the trail
        self.boxes = []                                 # (x0, x1, y0, y1, top)

    def merge_boxes(self, cfg):
        """Cover the columns with as few boxes as possible (greedy rectangles of equal height)."""
        q = np.round(self.height / QUANT).astype(np.int64)
        done = q == 0
        x0, y0, c = -cfg.tile_length / 2, -cfg.tile_width / 2, cfg.cell
        self.boxes = []
        for i in range(self.nx):
            for j in range(self.ny):
                if done[i, j]:
                    continue
                h = q[i, j]
                j1 = j + 1
                while j1 < self.ny and q[i, j1] == h and not done[i, j1]:
                    j1 += 1
                i1 = i + 1
                while i1 < self.nx and (q[i1, j:j1] == h).all() and not done[i1, j:j1].any():
                    i1 += 1
                done[i:i1, j:j1] = True
                self.boxes.append((x0 + i * c, x0 + i1 * c, y0 + j * c, y0 + j1 * c, h * QUANT))

    def rasterize(self, cfg):
        k = int(round(cfg.cell / RES))
        return np.repeat(np.repeat(self.height, k, axis=0), k, axis=1).astype(np.float32)


# ------------------------------------------------------------------ sections
class _Builder:
    """Working state while a trail is assembled section by section."""

    def __init__(self, tile, row, t, rng, cfg):
        self.tile, self.row, self.t, self.rng, self.cfg = tile, row, t, rng, cfg
        nx, ny = tile.nx, tile.ny
        self.elev_left = np.zeros(nx)       # height of the left (+y) / right half of the trail
        self.elev_right = np.zeros(nx)
        self.bump = np.zeros((nx, ny))      # added on top, per column
        self.solid = np.ones((nx, ny), dtype=bool)
        self.center = np.zeros(nx)          # centre line of the trail
        self.half_width = np.zeros(nx)
        self.path_y = np.full(nx, np.nan)   # where the robot should walk (nan = interpolate)
        self.boulders = []                  # (i0, i1, j0, j1, height)
        self.e = cfg.base_elevation

    def param(self, name):
        return _lerp(self.row.get(name, getattr(self.cfg, name)), self.t)

    def cells(self, metres, minimum=1):
        return max(minimum, int(round(metres / self.cfg.cell)))

    def set_elevation(self, i, e, shift_right=0):
        """From column i on the trail is at height e (the right half `shift_right` columns later)."""
        self.elev_left[i:] = e
        self.elev_right[max(i + shift_right, 0):] = e
        self.e = e


def _sec_flat(b, i0, i1):
    pass


def _block_noise(b, i0, i1, block, amplitude):
    rng = b.rng
    nbx, nby = (i1 - i0) // block + 1, b.tile.ny // block + 1
    noise = np.repeat(np.repeat(rng.uniform(0.0, amplitude, (nbx, nby)), block, axis=0), block, axis=1)
    b.bump[i0:i1, :] += noise[:i1 - i0, :b.tile.ny]


def _sec_rough(b, i0, i1):
    """Bumpy paving: 37.5 cm slabs at random heights."""
    _block_noise(b, i0, i1, 3, b.param("rough_height"))


def _sec_rubble(b, i0, i1):
    """(test only) smaller and taller bumps than `rough`."""
    _block_noise(b, i0, i1, 2, _lerp((0.03, 0.14), b.t))


def _steps(b, i0, i1, direction, regular=False):
    rng, cfg, t = b.rng, b.cfg, b.t
    lo, hi = cfg.elevation_range
    rise_max = b.param("step_rise")
    i = i0 + 2
    regular_run = b.cells(max(0.25, 1.5 * rise_max))
    while i < i1 - 2:
        rise = rise_max if regular else rise_max * rng.uniform(0.6, 1.0)
        sign = {"up": 1.0, "down": -1.0}.get(direction, rng.choice([-1.0, 1.0]))
        if b.e + sign * rise > hi or b.e + sign * rise < lo:
            sign = -sign
        # sometimes the edge of a step is broken: one half of the trail steps up earlier than the other
        shift = int(rng.choice([-2, -1, 1, 2])) if (not regular and rng.random() < 0.6 * t) else 0
        b.set_elevation(i, b.e + sign * rise, shift)
        run = regular_run if regular else b.cells(rng.uniform(_lerp((0.75, 0.375), t), _lerp((1.0, 0.625), t)))
        i += max(run, b.cells(1.6 * rise), abs(shift) + 1)


def _sec_steps(b, i0, i1):
    """Irregular rock steps, randomly up or down."""
    _steps(b, i0, i1, "mixed")


def _sec_steps_up(b, i0, i1):
    _steps(b, i0, i1, "up")


def _sec_steps_down(b, i0, i1):
    _steps(b, i0, i1, "down")


def _sec_stairs_up(b, i0, i1):
    """(test only) a regular staircase."""
    _steps(b, i0, i1, "up", regular=True)


def _sec_stairs_down(b, i0, i1):
    _steps(b, i0, i1, "down", regular=True)


def _sec_stones(b, i0, i1):
    """Stepping stones: only the stones are solid, they zigzag and (on higher levels) have gaps between them."""
    rng, tile, t = b.rng, b.tile, b.t
    half, zig = b.param("stone_half_width"), b.param("stone_zigzag")
    gap = int(round(b.param("stone_gap") / b.cfg.cell))
    a, end = i0 + 2, i1 - 2
    if end - a < 4 + 2 * gap:
        return
    b.solid[a:end, :] = False
    b.path_y[a:end] = np.nan
    i, side = a + gap, rng.choice([-1.0, 1.0])
    while i < end - gap:
        n = int(rng.integers(3, 6))
        if end - gap - (i + n) < 3 + gap:           # no room for another stone: this one is the last
            n = end - gap - i
        offset = side * zig * rng.uniform(0.5, 1.0)
        top = rng.uniform(0.0, 0.06) * t
        for k in range(i, i + n):
            on = np.abs(tile.y - (b.center[k] + offset)) <= half
            b.solid[k, on] = True
            b.bump[k, on] += top
            b.path_y[k] = b.center[k] + offset
        i += n + gap
        side = -side


def _sec_ridge(b, i0, i1):
    """A narrow ridge."""
    b.half_width[i0:i1] = b.param("ridge_half_width")


def _sec_boulders(b, i0, i1):
    """A wide stretch with boulders close to the line the robot walks."""
    rng, tile, cfg = b.rng, b.tile, b.cfg
    b.half_width[i0:i1] = np.maximum(b.half_width[i0:i1], 1.5)
    for _ in range(int(round(b.param("boulders")))):
        size = int(rng.integers(3, 6))
        if i1 - 2 - size <= i0 + 2:
            break
        ic = int(rng.integers(i0 + 2, i1 - 2 - size))
        side = rng.choice([-1.0, 1.0])
        clearance = b.param("boulder_clearance") + rng.uniform(0.0, 0.3)
        near = b.center[max(ic - 2, 0):ic + size + 2]
        face = (near.max() if side > 0 else near.min()) + side * clearance
        j0 = int(np.floor((min(face, face + side * size * cfg.cell) + cfg.tile_width / 2) / cfg.cell + 0.5))
        b.boulders.append((ic, ic + size, max(j0, 0), min(j0 + size, tile.ny), rng.uniform(0.5, 0.9)))


def _sec_logs(b, i0, i1):
    """(test only) logs lying across the trail."""
    rng, i = b.rng, i0 + 3
    while i < i1 - 3:
        thick = int(rng.integers(1, 3))
        b.bump[i:i + thick, :] += _lerp((0.04, 0.20), b.t) * rng.uniform(0.7, 1.0)
        i += thick + b.cells(rng.uniform(0.9, 1.4))


def _sec_terraces(b, i0, i1):
    """(test only) the left and the right half of the trail at different heights."""
    rng, tile, i = b.rng, b.tile, i0 + 2
    side = rng.choice([-1.0, 1.0])
    while i < i1 - 2:
        n = min(b.cells(rng.uniform(1.25, 1.75)), i1 - 2 - i)
        for k in range(i, i + n):
            b.bump[k, side * (tile.y - b.center[k]) > 0] += _lerp((0.04, 0.18), b.t)
        i += n
        side = -side


SECTIONS = {
    "flat": _sec_flat, "rough": _sec_rough, "steps": _sec_steps, "steps_up": _sec_steps_up,
    "steps_down": _sec_steps_down, "stones": _sec_stones, "ridge": _sec_ridge, "boulders": _sec_boulders,
    # only used by the held-out test trails
    "logs": _sec_logs, "terraces": _sec_terraces, "rubble": _sec_rubble,
    "stairs_up": _sec_stairs_up, "stairs_down": _sec_stairs_down,
}


def make_trail(row, t, rng, cfg=HikeCfg):
    """Generate one trail of the given row at difficulty t in [0, 1]."""
    tile = Tile(cfg)
    b = _Builder(tile, row, t, rng, cfg)
    x, nx = tile.x, tile.nx

    # ---- centre line: random swings left / right, smoothly joined
    hw = b.param("half_width")
    swing = b.param("winding")
    limit = cfg.tile_width / 2 - 1.5 - 0.3              # boulder sections are 1.5 m wide on each side
    knots_x = np.arange(-4.75, 5.3, 2.0)
    knots_y = np.zeros(knots_x.size)
    max_change = row.get("max_swing_change", 1.2 if swing <= 1.6 else 2.0)
    for k in range(1, knots_x.size):
        target = rng.uniform(-swing, swing)
        knots_y[k] = np.clip(np.clip(target, knots_y[k - 1] - max_change, knots_y[k - 1] + max_change), -limit, limit)
    seg = np.clip(np.searchsorted(knots_x, x) - 1, 0, knots_x.size - 2)
    u = np.clip((x - knots_x[seg]) / (knots_x[seg + 1] - knots_x[seg]), 0.0, 1.0)
    b.center = knots_y[seg] + (knots_y[seg + 1] - knots_y[seg]) * (1 - np.cos(np.pi * u)) / 2
    b.center = np.round(b.center / cfg.cell) * cfg.cell + cfg.cell / 2 * 0     # centre on a cell edge
    b.path_y[:] = b.center
    b.half_width[:] = hw
    b.set_elevation(0, cfg.base_elevation)

    # ---- sections between the start pad and the goal pad
    first = int(np.searchsorted(x, -4.75))
    last = int(np.searchsorted(x, 5.0))
    b.half_width[:first] = np.maximum(hw, 1.0)
    b.half_width[last:] = np.maximum(hw, 1.0)
    i = first
    while i < last:
        n = b.cells(rng.uniform(1.75, 3.0))
        if last - (i + n) < b.cells(1.5):
            n = last - i
        b.set_elevation(i, b.e)                         # squares off a broken step edge left by the last section
        SECTIONS[str(rng.choice(row["sections"]))](b, i, i + n)
        i += n
    b.set_elevation(last, b.e)

    # ---- assemble the column heights
    y = tile.y[None, :]
    on_trail = np.abs(y - b.center[:, None]) <= b.half_width[:, None] + 1e-9
    base = np.where(y >= b.center[:, None], b.elev_left[:, None], b.elev_right[:, None])
    height = np.where(on_trail & b.solid, base + b.bump, 0.0)
    ground = height.copy()
    for i0, i1, j0, j1, h in b.boulders:
        block = ground[i0:i1, j0:j1]
        if block.size:
            height[i0:i1, j0:j1] = np.where(block > 0, np.maximum(height[i0:i1, j0:j1], block.max() + h), 0.0)
    tile.height = np.round(height / QUANT) * QUANT

    known = ~np.isnan(b.path_y)
    tile.path_y = np.interp(np.arange(nx), np.nonzero(known)[0], b.path_y[known])
    tile.path_elev = np.minimum(b.elev_left, b.elev_right)
    tile.path_hw = b.half_width.copy()
    tile.merge_boxes(cfg)
    return tile


class Terrain:
    def __init__(self, rows, cfg=HikeCfg, seed=0):
        """
        Args:
            rows: list of row dicts with "name", "kind" ("flat" or "trail"), "weight" (share of
                  robots), "sections" and optional overrides of the HikeCfg difficulty ranges
            cfg: HikeCfg
            seed: seed of the random trail layouts
        """
        self.cfg = cfg
        self.rows = [dict(r) for r in rows]
        if self.rows[0]["kind"] != "flat":
            raise ValueError("the first terrain row must be flat")
        self.num_rows = len(self.rows)
        self.row_names = [r["name"] for r in self.rows]
        self.row_is_flat = np.array([r["kind"] == "flat" for r in self.rows])
        self.num_levels = 1 if self.row_is_flat.all() else cfg.num_levels

        rng = np.random.default_rng(seed)
        self.tiles = [Tile(cfg)]                                    # tile 0: flat ground
        self.tile_row, self.tile_level = [0], [0]
        self.tile_index = np.zeros((self.num_rows, self.num_levels), dtype=np.int64)
        for r, row in enumerate(self.rows):
            if row["kind"] == "flat":
                continue
            for level in range(self.num_levels):
                t = level / max(self.num_levels - 1, 1)
                self.tile_index[r, level] = len(self.tiles)
                self.tiles.append(make_trail(row, t, rng, cfg))
                self.tile_row.append(r)
                self.tile_level.append(level)
        self.num_tiles = len(self.tiles)

        self._raster = np.stack([tile.rasterize(cfg) for tile in self.tiles])
        self._inv_res = 1.0 / RES
        self._half_x, self._half_y = cfg.tile_length / 2, cfg.tile_width / 2
        self._max_ix, self._max_iy = self._raster.shape[1] - 1, self._raster.shape[2] - 1

        # path of every tile, sampled at the column centres
        self.path_x = self.tiles[0].x.copy()
        self.path_y = np.stack([tile.path_y for tile in self.tiles])
        self.path_elev = np.stack([tile.path_elev for tile in self.tiles])
        self.path_hw = np.stack([tile.path_hw for tile in self.tiles])
        # index of the path point `lookahead` metres (along the path) ahead of every column
        self.look_index = np.zeros_like(self.path_y, dtype=np.int64)
        nx = self.path_x.size
        for k in range(self.num_tiles):
            step = np.hypot(np.diff(self.path_x), np.diff(self.path_y[k]))
            arc = np.concatenate([[0.0], np.cumsum(step)])
            self.look_index[k] = np.clip(np.searchsorted(arc, arc + cfg.lookahead), 0, nx - 1)

    # ------------------------------------------------------------------ queries
    def height(self, tile, x, y):
        """Terrain height [m] at points (x, y) of the given tiles. tile broadcasts against x and y."""
        ix = np.clip(np.floor((np.asarray(x) + self._half_x) * self._inv_res), 0, self._max_ix).astype(np.int64)
        iy = np.clip(np.floor((np.asarray(y) + self._half_y) * self._inv_res), 0, self._max_iy).astype(np.int64)
        return self._raster[tile, ix, iy].astype(np.float64)

    def column(self, x):
        """Index of the path point at position x."""
        return np.clip(np.floor((np.asarray(x) + self._half_x) / self.cfg.cell), 0, self.path_x.size - 1).astype(np.int64)

    # ----------------------------------------------------------------- geometry
    def build_models(self, sim_cfg):
        """One compiled MjModel per tile (robot + ground + that trail). All have identical sizes."""
        most = max(len(tile.boxes) for tile in self.tiles)
        models = []
        for k, tile in enumerate(self.tiles):
            spec = mujoco.MjSpec.from_file(sim_cfg.model_path)
            world = spec.worldbody
            world.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                           material="grid", contype=0, conaffinity=1)
            row = self.rows[self.tile_row[k]]
            base = np.array(_COLORS.get(row["name"], _DEFAULT_COLOR))
            lo = self.cfg.elevation_range[0]
            for x0, x1, y0, y1, top in tile.boxes:
                # lighter with height, plus a little variation so neighbouring rocks can be told apart
                shade = 0.62 + 0.22 * np.clip((top - lo) / 2.0, 0.0, 1.0) + 0.09 * np.sin(37.0 * x0 + 91.0 * y0 + 53.0 * top)
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[(x0 + x1) / 2, (y0 + y1) / 2, top / 2],
                               size=[(x1 - x0) / 2, (y1 - y0) / 2, top / 2],
                               rgba=[*np.clip(base * shade / 0.8, 0.0, 1.0), 1], contype=0, conaffinity=1)
            for _ in range(most - len(tile.boxes)):
                # invisible filler far below the ground, so that all tile models have the same sizes
                # (number of geoms and collision tree nodes) and can share simulation buffers
                world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -50], size=[0.01, 0.01, 0.01],
                               rgba=[0, 0, 0, 0], contype=0, conaffinity=1)
            model = spec.compile()
            model.opt.timestep = sim_cfg.timestep
            models.append(model)
        sizes = {(m.nq, m.nv, m.nu, m.ngeom, m.nbvh, m.nsensordata) for m in models}
        if len(sizes) != 1:
            raise RuntimeError(f"tile models differ in size: {sizes}")
        return models
