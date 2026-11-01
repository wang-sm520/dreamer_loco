"""Shared static scene construction for Omni drone environments."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dynamic_geometry import dynamic_position_np


# ── geometry helpers used for obstacle placement ──────────────────────────────

def _obb_rect_corners2d(cx, cy, sx, sy, yaw):
    """XY corners of an OBB with center (cx, cy), size (sx, sy), and yaw (rad)."""
    hx, hy = 0.5 * float(sx), 0.5 * float(sy)
    c0 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    c0 *= np.array([hx, hy])
    co, si = np.cos(float(yaw)), np.sin(float(yaw))
    wx = co * c0[:, 0] - si * c0[:, 1] + float(cx)
    wy = si * c0[:, 0] + co * c0[:, 1] + float(cy)
    return np.column_stack([wx, wy])


def _corners_inside_arena2d(corners, lim_x, lim_y, eps=1e-3):
    return (np.max(np.abs(corners[:, 0])) <= float(lim_x) + eps
            and np.max(np.abs(corners[:, 1])) <= float(lim_y) + eps)


def _perimeter_inset_lims(terrain_cfg, perimeter_cfg):
    size = terrain_cfg.get('size', 30.0)
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        hx, hy = 0.5 * float(size[0]), 0.5 * float(size[1])
    else:
        hx = hy = 0.5 * float(size)
    if not bool((perimeter_cfg or {}).get('enabled', False)):
        return max(0.0, hx - 1e-3), max(0.0, hy - 1e-3)
    thickness = float((perimeter_cfg or {}).get('thickness', 0.3))
    return max(0.0, hx - 0.5 * thickness - 1e-3), max(0.0, hy - 0.5 * thickness - 1e-3)


def _obstacle_cheap_radius(sx, sy):
    return 0.5 * float(np.hypot(float(sx), float(sy)))


def _trap_wall_u_locals(W, t, d_left, d_right):
    """Local (lx, ly, sx, sy) for the 3 walls of a U-trap."""
    return [
        (0.0, t * 0.5, W + 2.0 * t, t),
        (-(W * 0.5 + t * 0.5), t + d_left * 0.5, t, d_left),
        (W * 0.5 + t * 0.5, t + d_right * 0.5, t, d_right),
    ]


def _trap_u_offsets_all_corners(cos_y, sin_y, W, t, d_left, d_right):
    """All OBB corner offsets (relative to trap center) after applying yaw."""
    rmat = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    chunks = []
    for lx, ly, sx, sy in _trap_wall_u_locals(W, t, d_left, d_right):
        hx, hy = 0.5 * sx, 0.5 * sy
        c0 = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
        c0 *= np.array([hx, hy])
        c0 += np.array([lx, ly])
        chunks.append((rmat @ c0.T).T)
    return np.vstack(chunks)


def _axis_interval_from_offsets(offsets, lim):
    """Valid range of trap center on one axis given corner offsets and arena half-size."""
    if offsets.size == 0:
        return -1e9, 1e9
    lo, hi = -1e9, 1e9
    for p in offsets.ravel():
        lo = max(lo, -float(lim) - float(p))
        hi = min(hi, float(lim) - float(p))
    return lo, hi


def _trap_walls_separate_from_obstacles(cx, cy, cos_y, sin_y, W, t, d_l, d_r, gap, obs_bounds):
    """Check that all 3 trap wall centers are far enough from existing obstacle disks."""
    for lx, ly, sx, sy in _trap_wall_u_locals(W, t, d_l, d_r):
        wx = cx + cos_y * lx - sin_y * ly
        wy = cy + sin_y * lx + cos_y * ly
        rw = _obstacle_cheap_radius(sx, sy)
        for item in obs_bounds:
            ox, oy = float(item[0]), float(item[1])
            osx, osy = float(item[3]), float(item[4])
            rd = _obstacle_cheap_radius(osx, osy)
            if np.hypot(wx - ox, wy - oy) <= rw + rd + float(gap):
                return False
    return True


def _trap_wall_centers_separate_from_disks(cx, cy, cos_y, sin_y, W, t, d_l, d_r, gap, disks):
    """Nav-compatible avoidance check: tests all 3 U-wall centers against a (cx,cy,r) disk list."""
    g = float(gap)
    for lx, ly, sx, sy in _trap_wall_u_locals(W, t, d_l, d_r):
        wx = float(cx + cos_y * lx - sin_y * ly)
        wy = float(cy + sin_y * lx + cos_y * ly)
        rw = _obstacle_cheap_radius(sx, sy)
        for ox, oy, rd in disks:
            if float(np.hypot(wx - ox, wy - oy)) <= rw + float(rd) + g:
                return False
    return True


@dataclass
class StaticObstacle:
    obstacle_id: str
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    prim_path: str = ''
    buffer_xy: float = 0.0
    yaw: float = 0.0


@dataclass
class DynamicObstacle:
    object_id: str
    prim_path: str
    kind: str
    shape: str
    size: tuple[float, float, float]
    collision_radius: float
    center: tuple[float, float, float]
    motion_type: str
    motion_axis: tuple[float, float, float]
    amplitude: float
    omega: float
    phase: float
    motion_radius: float
    lidar_visible: bool = True
    depth_visible: bool = True
    drone_collidable: bool = True


@dataclass
class SceneState:
    drone_pool: list = field(default_factory=list)
    drone_prim_paths: list = field(default_factory=list)
    env_positions: Any = None

    terrain_prim_paths: list[str] = field(default_factory=list)
    static_obstacles: list[StaticObstacle] = field(default_factory=list)
    dynamic_obstacles: list[DynamicObstacle] = field(default_factory=list)
    obstacle_bounds: list = field(default_factory=list)       # [(cx, cy, cz, sx, sy, sz, yaw)]
    trap_bounds: list = field(default_factory=list)           # [(cx, cy, yaw, W, D, t)]
    goal_visuals: list[str] = field(default_factory=list)

    terrain_cfg: dict = field(default_factory=dict)
    perimeter_cfg: dict = field(default_factory=dict)
    obstacles_cfg: dict = field(default_factory=dict)


def _bind_material_to_subtree(stage, root_prim, mat_prim):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material(mat_prim)
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if UsdPhysics.CollisionAPI.CanApply(prim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            try:
                UsdShade.MaterialBindingAPI(prim).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, 'physics'
                )
            except Exception:
                pass
        try:
            stack.extend(prim.GetChildren())
        except Exception:
            pass


def _set_collision_enabled_subtree(root_prim, enabled):
    value = bool(enabled)
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        try:
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr:
                attr.Set(value)
        except Exception:
            pass
        try:
            stack.extend(prim.GetChildren())
        except Exception:
            pass


def _resolve_drone_collision_root(prim_path: str):
    import omni.isaac.core.utils.prims as prim_utils

    for candidate in (f"{prim_path}/base_link/collisions", f"{prim_path}/base_link"):
        if not prim_utils.is_prim_path_valid(candidate):
            continue
        prim = prim_utils.get_prim_at_path(candidate)
        if prim is not None and prim.IsValid():
            return prim
    return None


def _setup_drone_collisions(prim_path: str):
    import omni.isaac.core.utils.prims as prim_utils

    if not prim_utils.is_prim_path_valid(str(prim_path)):
        return
    root = prim_utils.get_prim_at_path(str(prim_path))
    if root is None or (not root.IsValid()):
        return
    _set_collision_enabled_subtree(root, False)
    collision_root = _resolve_drone_collision_root(str(prim_path))
    if collision_root is None:
        _set_collision_enabled_subtree(root, True)
        return
    _set_collision_enabled_subtree(collision_root, True)


class SceneBuilder:

    def __init__(self, sim):
        self._sim = sim

    def build(
        self,
        *,
        num_drones,
        drone_model,
        device,
        terrain_cfg=None,
        perimeter_cfg=None,
        obstacles_cfg=None,
        create_goal_markers=True,
    ) -> SceneState:
        import torch
        from omni_drones.robots.robot import RobotBase

        terrain_cfg = dict(terrain_cfg or {})
        perimeter_cfg = dict(perimeter_cfg or {})
        obstacles_cfg = dict(obstacles_cfg or {})
        self._perimeter_cfg = perimeter_cfg
        np.random.seed(int(terrain_cfg.get('seed', 0)))

        self._spawn_lights()
        self._spawn_ground(int(num_drones), terrain_cfg)
        drones_root = self._ensure_drones_root()
        terrain_paths, static_obstacles = self._spawn_perimeter(terrain_cfg, perimeter_cfg)
        obs_paths, nav_obstacles, trap_bounds = self._spawn_static_obstacles(terrain_cfg, obstacles_cfg)
        _dynamic_paths, dynamic_obstacles = self._spawn_dynamic_obstacles(obstacles_cfg)
        terrain_paths.extend(obs_paths)
        static_obstacles.extend(nav_obstacles)
        drone_pool, drone_prim_paths = self._spawn_drones(
            num_drones=int(num_drones),
            drone_model=str(drone_model),
            device=str(device),
            drones_root=str(drones_root),
        )
        goal_visuals = self._spawn_goal_markers(int(num_drones)) if create_goal_markers else []

        env_positions = torch.zeros(
            (int(num_drones), 3),
            dtype=torch.float,
            device=torch.device(str(device)),
        )
        RobotBase._envs_positions = env_positions.unsqueeze(1)

        obstacle_bounds = []
        for obs in static_obstacles:
            if str(getattr(obs, 'kind', '')) == 'perimeter':
                continue
            cx, cy, cz = obs.center
            sx, sy, sz = obs.size
            yaw = float(getattr(obs, 'yaw', 0.0))
            obstacle_bounds.append((cx, cy, cz, sx, sy, sz, yaw))

        terrain_prim_paths = ['/World/ground']
        terrain_prim_paths.extend([p for p in terrain_paths if p and p != '/World/ground'])

        return SceneState(
            drone_pool=drone_pool,
            drone_prim_paths=drone_prim_paths,
            env_positions=env_positions,
            terrain_prim_paths=terrain_prim_paths,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            obstacle_bounds=obstacle_bounds,
            trap_bounds=trap_bounds,
            goal_visuals=goal_visuals,
            terrain_cfg=terrain_cfg,
            perimeter_cfg=perimeter_cfg,
            obstacles_cfg=obstacles_cfg,
        )

    def _spawn_lights(self):
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from omni_drones.utils.torch import euler_to_quaternion as _euler_to_quat

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = _euler_to_quat(torch.tensor([0.0, 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

    def _spawn_ground(self, num_drones, terrain_cfg):
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg

        tc = terrain_cfg or {}
        terrain_size = (float(tc.get('size', 30.0)), float(tc.get('size', 30.0)))
        cfg = TerrainImporterCfg(
            num_envs=max(1, int(num_drones)),
            prim_path="/World/ground",
            terrain_type="generator",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.22, 0.45),
                roughness=0.75,
                metallic=0.0,
            ),
            terrain_generator=TerrainGeneratorCfg(
                seed=int(tc.get('seed', 0)),
                size=terrain_size,
                border_width=float(tc.get('border_width', 5.0)),
                num_rows=1,
                num_cols=1,
                horizontal_scale=float(tc.get('horizontal_scale', 0.1)),
                vertical_scale=float(tc.get('vertical_scale', 0.005)),
                slope_threshold=float(tc.get('slope_threshold', 0.75)),
                use_cache=False,
                sub_terrains={
                    "ground": HfRandomUniformTerrainCfg(
                        size=terrain_size,
                        horizontal_scale=float(tc.get('horizontal_scale', 0.1)),
                        vertical_scale=float(tc.get('vertical_scale', 0.005)),
                        border_width=float(tc.get('border_width', 5.0)),
                        noise_range=(0.0, 0.0),
                        noise_step=float(tc.get('noise_step', 0.01)),
                    )
                },
            ),
            max_init_terrain_level=0,
            collision_group=-1,
            debug_vis=False,
        )
        return cfg.class_type(cfg)

    def _ensure_drones_root(self):
        import omni.isaac.core.utils.prims as prim_utils

        env_ns = '/World/envs'
        env_root = '/World/envs/env_0'
        drones_root = '/World/drones'
        for path in (env_ns, env_root, drones_root):
            if not prim_utils.is_prim_path_valid(path):
                prim_utils.define_prim(path)
        return drones_root

    def _spawn_perimeter(self, terrain_cfg, perimeter_cfg):
        import omni.isaac.core.utils.prims as prim_utils
        import isaaclab.sim as sim_utils

        if not bool(perimeter_cfg.get('enabled', False)):
            return [], []

        root = '/World/Perimeter'
        if not prim_utils.is_prim_path_valid(root):
            prim_utils.define_prim(root)

        half_x = float(terrain_cfg.get('size', 30.0)) * 0.5
        half_y = float(terrain_cfg.get('size', 30.0)) * 0.5
        thickness = float(perimeter_cfg.get('thickness', 0.3))
        height = float(perimeter_cfg.get('height', 4.0))
        color = tuple(float(x) for x in perimeter_cfg.get('color', (0.8, 0.8, 0.8)))
        walls = [
            ('north', 0.0, half_y, 2.0 * half_x + thickness, thickness, height),
            ('south', 0.0, -half_y, 2.0 * half_x + thickness, thickness, height),
            ('east', half_x, 0.0, thickness, 2.0 * half_y + thickness, height),
            ('west', -half_x, 0.0, thickness, 2.0 * half_y + thickness, height),
        ]

        obstacles = []
        for name, cx, cy, sx, sy, sz in walls:
            prim_path = f'{root}/{name}'
            spawn_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1),
            )
            spawn_cfg.func(prim_path, spawn_cfg, translation=(cx, cy, sz * 0.5))
            obstacles.append(
                StaticObstacle(
                    obstacle_id=f'perimeter_{name}',
                    kind='perimeter',
                    center=(float(cx), float(cy), float(sz * 0.5)),
                    size=(float(sx), float(sy), float(sz)),
                    prim_path=prim_path,
                    buffer_xy=0.0,
                )
            )
        return [root], obstacles

    def _spawn_static_obstacles(self, terrain_cfg, obstacles_cfg):
        import omni.isaac.core.utils.prims as prim_utils
        import isaaclab.sim as sim_utils

        roots = []
        obstacles = []
        trap_bounds = []
        lim_x, lim_y = _perimeter_inset_lims(terrain_cfg, getattr(self, '_perimeter_cfg', {}))
        if lim_x <= 0.0 or lim_y <= 0.0:
            size = float(terrain_cfg.get('size', 30.0))
            lim_x = lim_y = max(0.5, 0.5 * size - 1.0)
        collision_enabled = bool(obstacles_cfg.get('enable_physics_collision', True))
        default_gap = float(obstacles_cfg.get('obstacle_separation', 0.5))
        placement_disks = []

        # Order matches airdreamer_navigation: walls (skeleton) → traps → cuboids (clutter).
        # Cuboids skip wall disks so they may sit under/against long walls.

        walls_cfg = dict(obstacles_cfg.get('walls', {}) or {})
        wall_count = int(walls_cfg.get('count', 0))
        if wall_count > 0:
            root = '/World/WallObstacles'
            if not prim_utils.is_prim_path_valid(root):
                prim_utils.define_prim(root)
            roots.append(root)
            length = float(walls_cfg.get('length', 4.0))
            thickness = float(walls_cfg.get('thickness', 0.2))
            _whr = walls_cfg.get('height_range', None)
            _whf = float(walls_cfg.get('height', 3.0))
            if _whr is not None and isinstance(_whr, (list, tuple)) and len(_whr) >= 2:
                wall_height_range = (float(_whr[0]), float(_whr[1]))
            else:
                wall_height_range = (_whf, _whf)
            randomize_yaw = bool(walls_cfg.get('randomize_yaw', False))
            color = tuple(float(x) for x in walls_cfg.get('color', (0.8, 0.4, 0.1)))
            gap = float(walls_cfg.get('obstacle_separation', default_gap))
            map_span = float(min(lim_x, lim_y))
            spacing = 2.0 * np.sqrt(map_span ** 2 * 4.0 / max(wall_count, 1))
            wall_prev = []
            for idx in range(wall_count):
                curr_dist = spacing
                wall_h = float(np.random.uniform(wall_height_range[0], wall_height_range[1]))
                sx, sy = length, thickness
                if randomize_yaw:
                    yaw = float(np.random.uniform(-np.pi, np.pi))
                else:
                    yaw = (np.pi / 2.0) if np.random.randint(0, 2) else 0.0
                x, y = 0.0, 0.0
                for _ in range(300):
                    x = float(np.random.uniform(-map_span, map_span))
                    y = float(np.random.uniform(-map_span, map_span))
                    pt = np.array([x, y])
                    if all(np.linalg.norm(pt - p) > curr_dist for p in wall_prev):
                        if _corners_inside_arena2d(_obb_rect_corners2d(x, y, sx, sy, yaw), lim_x, lim_y):
                            wall_prev.append(pt)
                            break
                    curr_dist *= 0.95
                else:
                    x, y = 0.0, 0.0
                    for _ in range(80):
                        x = float(np.random.uniform(-map_span, map_span))
                        y = float(np.random.uniform(-map_span, map_span))
                        if _corners_inside_arena2d(_obb_rect_corners2d(x, y, sx, sy, yaw), lim_x, lim_y):
                            break
                    wall_prev.append(np.array([x, y]))
                qw = float(np.cos(yaw / 2.0))
                qz = float(np.sin(yaw / 2.0))
                prim_path = f'{root}/wall_{idx}'
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(sx, sy, wall_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.2),
                )
                spawn_cfg.func(
                    prim_path, spawn_cfg,
                    translation=(x, y, wall_h * 0.5),
                    orientation=(qw, 0.0, 0.0, qz),
                )
                placement_disks.append((float(x), float(y), _obstacle_cheap_radius(sx, sy)))
                obstacles.append(
                    StaticObstacle(
                        obstacle_id=f'wall_{idx}',
                        kind='wall',
                        center=(float(x), float(y), float(wall_h * 0.5)),
                        size=(float(sx), float(sy), float(wall_h)),
                        prim_path=prim_path,
                        buffer_xy=float(walls_cfg.get('buffer_xy', 0.0)),
                        yaw=float(yaw),
                    )
                )

        n_wall_disks = len(placement_disks)

        traps_cfg = dict(obstacles_cfg.get('traps', {}) or {})
        trap_count = int(traps_cfg.get('count', 0))
        if trap_count > 0:
            trap_obs, trap_bounds = self._spawn_trap_obstacles(
                sim_utils,
                prim_utils,
                lim_x,
                lim_y,
                traps_cfg,
                obstacles,
                placement_disks,
                collision_enabled,
                default_gap,
            )
            obstacles.extend(trap_obs)
            if trap_obs and '/World/TrapObstacles' not in roots:
                roots.append('/World/TrapObstacles')

        cuboids_cfg = dict(obstacles_cfg.get('cuboids', {}) or {})
        cuboid_count = int(cuboids_cfg.get('count', 0))
        if cuboid_count > 0:
            root = '/World/CuboidObstacles'
            if not prim_utils.is_prim_path_valid(root):
                prim_utils.define_prim(root)
            roots.append(root)
            width_range = list(cuboids_cfg.get('width_range', [0.25, 1.0]))
            height_range = list(cuboids_cfg.get(
                'height_range',
                [float(cuboids_cfg.get('height', 3.0)), float(cuboids_cfg.get('height', 3.0))],
            ))
            color = tuple(float(x) for x in cuboids_cfg.get('color', (0.0, 1.0, 0.0)))
            gap = float(cuboids_cfg.get('obstacle_separation', default_gap))
            w_max_fit = 2.0 * min(lim_x, lim_y) - 1e-2
            for idx in range(cuboid_count):
                # Avoidance disks: skip wall entries so cuboids may sit against walls.
                avoidance = placement_disks[n_wall_disks:]
                x, y = 0.0, 0.0
                placed = False
                w_try = None
                width = float(width_range[0])
                height = float(height_range[0])
                for pre in range(4):
                    if pre == 0:
                        width = float(np.random.uniform(width_range[0], width_range[1]))
                    elif pre == 1:
                        width = float(width_range[0])
                    elif pre == 2:
                        width = float(0.5 * (width_range[0] + width_range[1]))
                    else:
                        width = float(min(w_max_fit, max(width_range[0], 0.6 * w_max_fit)))
                    width = min(width, w_max_fit)
                    w_try = width
                    height = float(np.random.uniform(height_range[0], height_range[1]))
                    r_c = _obstacle_cheap_radius(width, width)
                    hw = 0.5 * width
                    rx = max(0.0, lim_x - hw)
                    ry = max(0.0, lim_y - hw)
                    disks = list(avoidance)
                    for _ in range(500):
                        x = float(np.random.uniform(-rx, rx)) if rx > 0 else 0.0
                        y = float(np.random.uniform(-ry, ry)) if ry > 0 else 0.0
                        if disks:
                            if not all(np.hypot(x - ox, y - oy) > r_c + rd + gap
                                       for (ox, oy, rd) in disks):
                                continue
                        if not _corners_inside_arena2d(
                                _obb_rect_corners2d(x, y, width, width, 0.0), lim_x, lim_y):
                            continue
                        placed = True
                        break
                    if placed:
                        break
                if not placed:
                    width = min(
                        float(w_try if w_try is not None else width_range[0]),
                        w_max_fit,
                    )
                    height = float(np.random.uniform(height_range[0], height_range[1]))
                    r_c = _obstacle_cheap_radius(width, width)
                    disks = list(avoidance)
                    for _ in range(120):
                        x = float(np.random.uniform(-lim_x, lim_x))
                        y = float(np.random.uniform(-lim_y, lim_y))
                        if not _corners_inside_arena2d(
                                _obb_rect_corners2d(x, y, width, width, 0.0), lim_x, lim_y):
                            continue
                        if all(np.hypot(x - ox, y - oy) > r_c + rd + 0.1
                               for (ox, oy, rd) in disks):
                            placed = True
                            break
                    if not placed:
                        x = y = 0.0
                        print(
                            f'[Obstacles] cuboid_{idx}: loose placement at origin '
                            f'(width={width:.2f}; may overlap).'
                        )
                yaw = 0.0
                prim_path = f'{root}/cuboid_{idx}'
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(width, width, height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.2),
                )
                spawn_cfg.func(prim_path, spawn_cfg, translation=(x, y, height * 0.5))
                placement_disks.append((float(x), float(y), _obstacle_cheap_radius(width, width)))
                obstacles.append(
                    StaticObstacle(
                        obstacle_id=f'cuboid_{idx}',
                        kind='cuboid',
                        center=(float(x), float(y), float(height * 0.5)),
                        size=(float(width), float(width), float(height)),
                        prim_path=prim_path,
                        buffer_xy=float(cuboids_cfg.get('buffer_xy', 0.0)),
                        yaw=float(yaw),
                    )
                )

        return roots, obstacles, trap_bounds

    def _spawn_trap_obstacles(
            self,
            sim_utils,
            prim_utils,
            lim_x,
            lim_y,
            traps_cfg,
            existing_obstacles,
            placement_disks,
            collision_enabled,
            default_gap):
        """Spawn U-shaped trap obstacles (3 walls each)."""
        count = int(traps_cfg.get('count', 0))
        W_range = list(traps_cfg.get('opening_width_range', [2.0, 4.0]))
        D_range = list(traps_cfg.get('arm_depth_range', traps_cfg.get('depth_range', [1.0, 2.0])))
        H_range = list(traps_cfg.get('height_range', [2.0, 3.0]))
        T_range = list(traps_cfg.get('thickness_range', [0.15, 0.3]))
        wr = (float(W_range[0]), float(W_range[1]))
        dr = (float(D_range[0]), float(D_range[1]))
        hr = (float(H_range[0]), float(H_range[1]))
        tr = (float(T_range[0]), float(T_range[1]))
        asym = bool(traps_cfg.get('asymmetric_arms', False))
        rand_yaw = bool(traps_cfg.get('randomize_yaw', True))
        color = tuple(float(x) for x in traps_cfg.get('color', (0.1, 0.4, 0.8)))
        gap = float(traps_cfg.get('obstacle_separation', default_gap))

        root = '/World/TrapObstacles'
        if not prim_utils.is_prim_path_valid(root):
            prim_utils.define_prim(root)

        new_obstacles = []
        trap_bounds = []

        for i in range(count):
            placed = False
            best = None
            for attempt in range(4):
                if attempt == 0:
                    W = float(np.random.uniform(wr[0], wr[1]))
                    t = float(np.random.uniform(tr[0], tr[1]))
                    D_l = float(np.random.uniform(dr[0], dr[1]))
                elif attempt == 1:
                    W, t, D_l = wr[0], tr[0], dr[0]
                elif attempt == 2:
                    W = 0.5 * (wr[0] + wr[1])
                    t = 0.5 * (tr[0] + tr[1])
                    D_l = 0.5 * (dr[0] + dr[1])
                else:
                    W = max(wr[0], 0.6 * wr[1])
                    t = max(tr[0], 0.6 * tr[1])
                    D_l = max(dr[0], 0.6 * dr[1])
                D_r = float(np.random.uniform(dr[0], dr[1])) if asym else D_l
                H = float(np.random.uniform(hr[0], hr[1]))
                if rand_yaw:
                    yaw = float(np.random.uniform(-np.pi, np.pi))
                else:
                    yaw = float(np.random.choice([0.0, np.pi / 2.0, np.pi, 3 * np.pi / 2.0]))
                cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
                u_off = _trap_u_offsets_all_corners(cos_y, sin_y, W, t, D_l, D_r)
                cx_lo, cx_hi = _axis_interval_from_offsets(u_off[:, 0], lim_x)
                cy_lo, cy_hi = _axis_interval_from_offsets(u_off[:, 1], lim_y)
                if cx_lo > cx_hi or cy_lo > cy_hi:
                    continue
                best = (W, t, D_l, D_r, H, yaw, cos_y, sin_y, u_off, cx_lo, cx_hi, cy_lo, cy_hi)
                disks = list(placement_disks)
                for _ in range(500):
                    cx = float(np.random.uniform(cx_lo, cx_hi))
                    cy = float(np.random.uniform(cy_lo, cy_hi))
                    if not _trap_wall_centers_separate_from_disks(
                            cx, cy, cos_y, sin_y, W, t, D_l, D_r, gap, disks):
                        continue
                    corn = u_off + np.array([cx, cy])
                    if not _corners_inside_arena2d(corn, lim_x, lim_y):
                        continue
                    placed = True
                    break
                if placed:
                    break

            if not placed and best is not None:
                W, t, D_l, D_r, H, yaw, cos_y, sin_y, u_off, cx_lo, cx_hi, cy_lo, cy_hi = best
                # Second pass with relaxed gap=0.1 before falling back to centre.
                second_pass = False
                disks = list(placement_disks)
                for _ in range(120):
                    cx = float(np.random.uniform(cx_lo, cx_hi))
                    cy = float(np.random.uniform(cy_lo, cy_hi))
                    if not _trap_wall_centers_separate_from_disks(
                            cx, cy, cos_y, sin_y, W, t, D_l, D_r, 0.1, disks):
                        continue
                    corn = u_off + np.array([cx, cy])
                    if not _corners_inside_arena2d(corn, lim_x, lim_y):
                        continue
                    second_pass = True
                    break
                if not second_pass:
                    cx = 0.5 * (cx_lo + cx_hi)
                    cy = 0.5 * (cy_lo + cy_hi)
                    print(
                        f'[Obstacles] trap_{i}: no separated placement; '
                        f'using centre of last U footprint (may overlap).'
                    )
                placed = True
            elif not placed:
                print(f'[Obstacles] trap_{i}: skipped (no valid geometry).')
                continue

            qw = float(np.cos(yaw / 2.0))
            qz = float(np.sin(yaw / 2.0))
            walls_local = _trap_wall_u_locals(W, t, D_l, D_r)
            for j, (lx, ly, sx, sy) in enumerate(walls_local):
                wx = float(cx + cos_y * lx - sin_y * ly)
                wy = float(cy + sin_y * lx + cos_y * ly)
                prim_path = f'{root}/trap_{i}_wall_{j}'
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(sx, sy, H),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True, kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.2),
                )
                spawn_cfg.func(
                    prim_path, spawn_cfg,
                    translation=(wx, wy, H * 0.5),
                    orientation=(qw, 0.0, 0.0, qz),
                )
                new_obstacles.append(
                    StaticObstacle(
                        obstacle_id=f'trap_{i}_wall_{j}',
                        kind='trap_wall',
                        center=(wx, wy, float(H * 0.5)),
                        size=(float(sx), float(sy), float(H)),
                        prim_path=prim_path,
                        yaw=float(yaw),
                    )
                )
                placement_disks.append((float(wx), float(wy), _obstacle_cheap_radius(sx, sy)))

            D_trap = max(D_l, D_r)
            trap_bounds.append((float(cx), float(cy), float(yaw), float(W), float(D_trap), float(t)))

        if new_obstacles and not prim_utils.is_prim_path_valid(root):
            prim_utils.define_prim(root)
        print(f'[Obstacles] Spawned {len(trap_bounds)} trap obstacles.')
        return new_obstacles, trap_bounds

    def _spawn_dynamic_obstacles(self, obstacles_cfg):
        import omni.isaac.core.utils.prims as prim_utils
        import isaaclab.sim as sim_utils

        dynamic_cfg = dict(obstacles_cfg.get('dynamic', {}) or {})
        enabled = bool(dynamic_cfg.get('enabled', False))
        raw_items = dynamic_cfg.get('items', []) or []
        items = []
        if isinstance(raw_items, dict):
            for object_id, payload in raw_items.items():
                item = dict(payload or {})
                item.setdefault('object_id', str(object_id))
                items.append(item)
        else:
            for payload in list(raw_items):
                item = dict(payload or {})
                items.append(item)
        if (not enabled) or (not items):
            return [], []

        root = '/World/DynamicObstacles'
        if not prim_utils.is_prim_path_valid(root):
            prim_utils.define_prim(root)

        collision_enabled = bool(dynamic_cfg.get('enable_physics_collision', False))
        speed_scale = float(dynamic_cfg.get('speed_scale', 1.0))
        obstacles = []
        for idx, raw in enumerate(items):
            item = dict(raw or {})
            motion = dict(item.get('motion', {}) or {})

            object_id = str(item.get('object_id', f'dyn_{idx}'))
            kind = str(item.get('kind', 'dynamic_obstacle'))
            shape = str(item.get('shape', 'box')).lower()
            center = tuple(float(x) for x in motion.get('center', item.get('center', (0.0, 0.0, 1.2))))
            motion_type = str(motion.get('type', 'line')).lower()
            motion_axis = tuple(float(x) for x in motion.get('direction', motion.get('axis', (1.0, 0.0, 0.0))))
            amplitude = float(motion.get('amplitude', 0.0))
            omega = float(motion.get('omega', motion.get('speed', 0.0))) * speed_scale
            phase = float(motion.get('phase', 0.0))
            motion_radius = float(motion.get('radius', amplitude))
            color = tuple(float(x) for x in item.get('color', (1.0, 0.6, 0.1)))
            lidar_visible = bool(item.get('lidar_visible', True))
            depth_visible = bool(item.get('depth_visible', True))
            drone_collidable = bool(item.get('drone_collidable', True))

            prim_path = f'{root}/{object_id}'
            if shape == 'box':
                raw_size = list(item.get('size', [0.7, 0.7, 0.7]))
                while len(raw_size) < 3:
                    raw_size.append(raw_size[-1] if raw_size else 0.6)
                sx, sy, sz = [max(1e-3, float(v)) for v in raw_size[:3]]
                size = (sx, sy, sz)
                collision_radius = float(item.get('collision_radius', 0.5 * np.linalg.norm(size)))
                spawn_cfg = sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1),
                )
            else:
                shape = 'sphere'
                radius_value = item.get('radius', item.get('size', 0.35))
                if isinstance(radius_value, (list, tuple)):
                    raw_radius = [max(1e-3, float(v)) for v in radius_value[:3]]
                    radius = 0.5 * max(raw_radius) if raw_radius else 0.35
                else:
                    radius = max(1e-3, float(radius_value))
                size = (2.0 * radius, 2.0 * radius, 2.0 * radius)
                collision_radius = float(item.get('collision_radius', radius))
                spawn_cfg = sim_utils.SphereCfg(
                    radius=radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision_enabled),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.1),
                )

            p0 = dynamic_position_np(
                center=center,
                motion_type=motion_type,
                motion_axis=motion_axis,
                amplitude=amplitude,
                omega=omega,
                phase=phase,
                motion_radius=motion_radius,
                t=0.0,
            )
            spawn_cfg.func(prim_path, spawn_cfg, translation=(float(p0[0]), float(p0[1]), float(p0[2])))
            obstacles.append(
                DynamicObstacle(
                    object_id=object_id,
                    prim_path=prim_path,
                    kind=kind,
                    shape=shape,
                    size=size,
                    collision_radius=collision_radius,
                    center=center,
                    motion_type=motion_type,
                    motion_axis=motion_axis,
                    amplitude=amplitude,
                    omega=omega,
                    phase=phase,
                    motion_radius=motion_radius,
                    lidar_visible=lidar_visible,
                    depth_visible=depth_visible,
                    drone_collidable=drone_collidable,
                )
            )
        return [root], obstacles

    def _sample_obstacle_centers(self, count, map_half):
        if count <= 0:
            return []
        spacing = 2.0 * np.sqrt(map_half * map_half * 4.0 / max(count, 1))
        positions = []
        accepted = []
        for _ in range(int(count)):
            curr_dist = spacing
            x, y = 0.0, 0.0
            for _ in range(300):
                x = float(np.random.uniform(-map_half, map_half))
                y = float(np.random.uniform(-map_half, map_half))
                pt = np.array([x, y], np.float32)
                if all(np.linalg.norm(pt - prev) > curr_dist for prev in accepted):
                    accepted.append(pt)
                    break
                curr_dist *= 0.95
            else:
                accepted.append(np.array([x, y], np.float32))
            positions.append((x, y))
        return positions

    def _sample_obb_center(self, sx, sy, yaw, lim_x, lim_y, accepted_disks, gap):
        corners0 = _obb_rect_corners2d(0.0, 0.0, sx, sy, yaw)
        cx_lo, cx_hi = _axis_interval_from_offsets(corners0[:, 0], lim_x)
        cy_lo, cy_hi = _axis_interval_from_offsets(corners0[:, 1], lim_y)
        if cx_lo > cx_hi or cy_lo > cy_hi:
            return 0.0, 0.0
        radius = _obstacle_cheap_radius(sx, sy)
        for _ in range(500):
            x = float(np.random.uniform(cx_lo, cx_hi))
            y = float(np.random.uniform(cy_lo, cy_hi))
            if any(np.hypot(x - ox, y - oy) <= radius + float(orad) + float(gap)
                   for ox, oy, orad in accepted_disks):
                continue
            corners = corners0 + np.array([x, y], dtype=np.float64)
            if not _corners_inside_arena2d(corners, lim_x, lim_y):
                continue
            return x, y

        best = None
        best_score = -float('inf')
        for _ in range(80):
            x = float(np.random.uniform(cx_lo, cx_hi))
            y = float(np.random.uniform(cy_lo, cy_hi))
            if not accepted_disks:
                return x, y
            score = min(
                float(np.hypot(x - ox, y - oy) - radius - float(orad))
                for ox, oy, orad in accepted_disks
            )
            if score > best_score:
                best_score = score
                best = (x, y)
        if best is not None:
            return best
        return 0.5 * (cx_lo + cx_hi), 0.5 * (cy_lo + cy_hi)

    def _spawn_drones(self, *, num_drones, drone_model, device, drones_root):
        import omni.isaac.core.utils.prims as prim_utils
        import isaaclab.sim as sim_utils
        from .drone_factory import make_drone

        drone_pool = []
        drone_prim_paths = []
        for idx in range(int(num_drones)):
            drone_name = f'{str(drone_model)}_{int(idx)}'
            drone_path = f'{drones_root}/drone_{int(idx)}'
            drone, _ = make_drone(drone_model, None, str(device), name=drone_name)
            drone.spawn(translations=[(0.0, 0.0, 2.0)], prim_paths=[drone_path])
            _setup_drone_collisions(drone_path)

            try:
                no_bounce = sim_utils.RigidBodyMaterialCfg(
                    restitution=0.0,
                    restitution_combine_mode="min",
                    static_friction=0.5,
                    dynamic_friction=0.5,
                )
                no_bounce.func(f'{drone_path}/physics_material', no_bounce)
                stage = prim_utils.get_current_stage()
                mat_prim = stage.GetPrimAtPath(f'{drone_path}/physics_material')
                if mat_prim.IsValid():
                    bind_root = _resolve_drone_collision_root(drone_path)
                    if bind_root is None:
                        bind_root = prim_utils.get_prim_at_path(drone_path)
                    _bind_material_to_subtree(stage, bind_root, mat_prim)
            except Exception:
                pass

            drone_pool.append(drone)
            drone_prim_paths.append(drone_path)
        return drone_pool, drone_prim_paths

    def _spawn_goal_markers(self, num_drones):
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import Gf, UsdGeom

        goal_visuals = []
        goal_radius = 0.2
        for idx in range(int(num_drones)):
            prim_path = f"/World/GoalMarker_{idx}"
            if not prim_utils.is_prim_path_valid(prim_path):
                prim_utils.create_prim(
                    prim_path,
                    "Sphere",
                    translation=(0.0, 0.0, -100.0),
                    scale=(goal_radius, goal_radius, goal_radius),
                )
            prim = prim_utils.get_prim_at_path(prim_path)
            if prim and prim.IsValid():
                try:
                    gprim = UsdGeom.Gprim(prim)
                    gprim.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.1, 0.1)])
                except Exception:
                    pass
            goal_visuals.append(prim_path)
        return goal_visuals
