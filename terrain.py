"""
terrain.py

Author: Usman Siddiqi

Procedural terrain for the biped: flat ground plus staircases of increasing
difficulty. The Terrain class does two jobs that must agree with each other:

  1. build the collision geometry (boxes) inside the MuJoCo model
  2. answer "how high is the ground at (x, y)?" for thousands of points at once
     (used for the robot's height scan, rewards and resets) with plain numpy

Layout: tile (row = terrain type, column = level) has its centre at
    x = (level + 0.5) * tile_size,   y = (row + 0.5) * tile_size

Stairs are square "pyramids" so that the robot meets steps whichever way it
walks away from the tile centre:

  stairs_up    a pit: the centre platform is at z = 0 and every ring further
               out is one step higher (the robot climbs out)
  stairs_down  a ziggurat: the centre platform is the highest point and the
               rings step down to z = 0 (the robot walks down)

Flat ground is simply the infinite ground plane, so row 0 must be "flat":
robots there can wander off towards -y forever.
"""

import numpy as np
import mujoco

FLAT, STAIRS_UP, STAIRS_DOWN = 0, 1, 2
_KINDS = {"flat": FLAT, "stairs_up": STAIRS_UP, "stairs_down": STAIRS_DOWN}


class Terrain:
    def __init__(self, cfg, flat_only=False):
        self.cfg = cfg
        types = ["flat"] if flat_only else list(cfg.types)
        if types[0] != "flat":
            raise ValueError("the first terrain type must be 'flat'")
        self.kinds = np.array([_KINDS[t] for t in types], dtype=np.int64)
        self.num_rows = len(types)
        self.num_levels = 1 if flat_only else cfg.num_levels
        self.tile = float(cfg.tile_size)
        self.platform = float(cfg.platform_half)
        self.run = float(cfg.step_run)
        self.num_steps = int(cfg.num_steps)
        self.step_heights = np.linspace(cfg.step_height_min, cfg.step_height_max, self.num_levels)
        outer = self.platform + self.num_steps * self.run
        if outer + 0.5 > self.tile / 2 or cfg.success_distance > self.tile / 2 - 0.5:
            raise ValueError("tile_size is too small for the staircase")
        if cfg.success_distance < outer + 0.3:
            raise ValueError("success_distance must lie beyond the last step")

    # ------------------------------------------------------------------ queries
    def tile_center(self, row, level):
        """Centre (x, y) of tiles; row/level can be arrays."""
        return (np.asarray(level) + 0.5) * self.tile, (np.asarray(row) + 0.5) * self.tile

    def height(self, x, y):
        """Terrain height [m] at world points (x, y). Works on arrays of any shape."""
        if self.num_rows == 1:
            return np.zeros(np.shape(x), dtype=np.float64)
        level = np.clip(np.floor(x / self.tile), 0, self.num_levels - 1).astype(np.int64)
        row = np.clip(np.floor(y / self.tile), 0, self.num_rows - 1).astype(np.int64)
        dx = np.abs(x - (level + 0.5) * self.tile)
        dy = np.abs(y - (row + 0.5) * self.tile)
        dist = np.maximum(dx, dy)                       # square rings -> Chebyshev distance
        k = np.clip(np.ceil((dist - self.platform) / self.run), 0, self.num_steps)
        kind = self.kinds[row]
        steps = np.where(kind == STAIRS_UP, k, np.where(kind == STAIRS_DOWN, self.num_steps - k, 0.0))
        return steps * self.step_heights[level]

    def distance_from_center(self, x, y, row, level):
        cx, cy = self.tile_center(row, level)
        return np.maximum(np.abs(x - cx), np.abs(y - cy))

    # ----------------------------------------------------------------- geometry
    def add_to_spec(self, spec):
        """Add the ground plane and all stair boxes to a mujoco.MjSpec."""
        world = spec.worldbody
        world.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
            material="grid", contype=0, conaffinity=1,
        )
        half = self.tile / 2
        for row in range(self.num_rows):
            kind = self.kinds[row]
            if kind == FLAT:
                continue
            for level in range(self.num_levels):
                cx, cy = self.tile_center(row, level)
                h = self.step_heights[level]
                shade = 0.55 + 0.25 * level / max(self.num_levels - 1, 1)
                for k in range(1, self.num_steps + 1):
                    tint = 0.06 * (k % 2)
                    if kind == STAIRS_UP:
                        rgba = [shade * 0.75 - tint, shade * 0.85 - tint, shade - tint, 1]
                        inner = self.platform + (k - 1) * self.run     # this step starts here
                        top = k * h
                        # four slabs (+x, -x, +y, -y) reaching from the step edge to the tile border
                        mid, ext = (inner + half) / 2, (half - inner) / 2
                        slabs = [
                            ([cx + mid, cy], [ext, half]), ([cx - mid, cy], [ext, half]),
                            ([cx, cy + mid], [half, ext]), ([cx, cy - mid], [half, ext]),
                        ]
                        for (px, py), (sx, sy) in slabs:
                            self._add_box(world, [px, py, top / 2], [sx, sy, top / 2], rgba)
                    else:  # STAIRS_DOWN: stacked, shrinking squares
                        rgba = [shade - tint, shade * 0.85 - tint, shade * 0.7 - tint, 1]
                        q = self.num_steps - k + 1                      # height index of this slab
                        size = self.platform + (k - 1) * self.run
                        top = q * h
                        self._add_box(world, [cx, cy, top / 2], [size, size, top / 2], rgba)

    @staticmethod
    def _add_box(world, pos, size, rgba):
        world.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba,
            contype=0, conaffinity=1,
        )


def build_model(sim_cfg, terrain):
    """Load the robot MJCF, add the terrain and compile to an MjModel."""
    spec = mujoco.MjSpec.from_file(sim_cfg.model_path)
    terrain.add_to_spec(spec)
    model = spec.compile()
    model.opt.timestep = sim_cfg.timestep
    return model
