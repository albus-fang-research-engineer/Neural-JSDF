import numpy as np
import trimesh

def mesh_to_2d_maze(
    mesh_path: str,
    resolution: float = 0.02,      # meters per cell
    pad: float = 0.25,             # meters padding around mesh bounds
    up_axis: str = "z",            # axis treated as "height" for projection
    fill: bool = True,             # fill interiors (good if mesh forms closed wall outlines in 2D)
    invert: bool = False,          # if True: walls=0 free=1 (some planners want that)
    wall_value: int = 1,
    free_value: int = 0,
    robot_radius: float | None = None,  # meters; if set, inflate walls by this radius
):
    """
    Convert a mesh (obj/stl/ply/etc.) into a 2D numpy occupancy grid.
    Output: grid (H,W) uint8, and a dict with world<->grid transforms.

    Convention (default):
      - wall/occupied = 1
      - free          = 0
    """

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    # ---- project vertices to 2D plane ----
    verts = mesh.vertices.copy()

    axis_map = {"x": 0, "y": 1, "z": 2}
    up = axis_map[up_axis.lower()]
    keep = [i for i in [0, 1, 2] if i != up]  # 2 axes kept for 2D

    verts_2d = verts[:, keep]  # (N,2)

    # ---- compute bounds + grid size ----
    vmin = verts_2d.min(axis=0) - pad
    vmax = verts_2d.max(axis=0) + pad
    size = vmax - vmin

    W = int(np.ceil(size[0] / resolution))
    H = int(np.ceil(size[1] / resolution))

    # world -> grid: (x,y) -> (col,row)
    def world_to_grid(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        ij = (xy - vmin[None, :]) / resolution
        col = np.floor(ij[:, 0]).astype(int)
        row = np.floor(ij[:, 1]).astype(int)
        return np.stack([row, col], axis=1)

    def grid_to_world(rc: np.ndarray) -> np.ndarray:
        rc = np.asarray(rc, dtype=np.float64)
        row, col = rc[:, 0], rc[:, 1]
        x = vmin[0] + (col + 0.5) * resolution
        y = vmin[1] + (row + 0.5) * resolution
        return np.stack([x, y], axis=1)

    # ---- build a 2D "mesh" in the projected plane ----
    # We can use the mesh polygons by projecting triangles.
    # Create a 2D Trimesh-like object with planar triangles for rasterization via contains.
    faces = mesh.faces
    tri_2d = verts_2d[faces]  # (F,3,2)

    # ---- rasterize ----
    grid = np.full((H, W), free_value, dtype=np.uint8)

    # sample points at cell centers and test if they are inside projected triangle union
    # Strategy:
    # - Build a 2D polygon soup by using triangles as shapely polygons (via trimesh.path.polygons)
    # - Then fill. This is the most robust if you truly want "filled walls".
    #
    # Practical approach without shapely:
    # - Mark triangle edges as occupied using line rasterization (good for "thin walls")
    # - Optionally fill closed regions via flood fill from boundary.
    #
    # Here: edge rasterization + optional fill.

    def rasterize_line(p0_rc, p1_rc):
        """Bresenham-ish integer line in (row,col)."""
        r0, c0 = p0_rc
        r1, c1 = p1_rc
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 >= r0 else -1
        sc = 1 if c1 >= c0 else -1
        if dc > dr:
            err = dc / 2
            r = r0
            for c in range(c0, c1 + sc, sc):
                if 0 <= r < H and 0 <= c < W:
                    grid[r, c] = wall_value
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
        else:
            err = dr / 2
            c = c0
            for r in range(r0, r1 + sr, sr):
                if 0 <= r < H and 0 <= c < W:
                    grid[r, c] = wall_value
                err -= dc
                if err < 0:
                    c += sc
                    err += dr

    # mark all triangle edges as walls
    for t in tri_2d:
        pts = np.array(t, dtype=np.float64)  # (3,2)
        rc = world_to_grid(pts)              # (3,2) row,col
        rasterize_line(rc[0], rc[1])
        rasterize_line(rc[1], rc[2])
        rasterize_line(rc[2], rc[0])

    if fill:
        # Flood fill from border to mark "outside free space", then invert to fill enclosed walls.
        # This assumes your mesh outlines form closed barriers in 2D.
        from collections import deque

        outside = np.zeros_like(grid, dtype=bool)
        q = deque()

        # enqueue all border free cells
        for c in range(W):
            if grid[0, c] == free_value:
                outside[0, c] = True; q.append((0, c))
            if grid[H-1, c] == free_value:
                outside[H-1, c] = True; q.append((H-1, c))
        for r in range(H):
            if grid[r, 0] == free_value:
                outside[r, 0] = True; q.append((r, 0))
            if grid[r, W-1] == free_value:
                outside[r, W-1] = True; q.append((r, W-1))

        nbrs = [(1,0), (-1,0), (0,1), (0,-1)]
        while q:
            r, c = q.popleft()
            for dr, dc in nbrs:
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W and (not outside[rr, cc]) and grid[rr, cc] == free_value:
                    outside[rr, cc] = True
                    q.append((rr, cc))

        # anything not reachable from outside and not already wall -> treat as wall (filled)
        enclosed = (~outside) & (grid == free_value)
        grid[enclosed] = wall_value

    if robot_radius is not None and robot_radius > 0:
        # Inflate walls by robot radius using binary dilation
        try:
            from scipy.ndimage import binary_dilation
            r_cells = int(np.ceil(robot_radius / resolution))
            if r_cells > 0:
                occ = grid == wall_value
                # circular-ish structuring element
                yy, xx = np.ogrid[-r_cells:r_cells+1, -r_cells:r_cells+1]
                selem = (xx*xx + yy*yy) <= (r_cells*r_cells)
                occ2 = binary_dilation(occ, structure=selem)
                grid = np.where(occ2, wall_value, free_value).astype(np.uint8)
        except ImportError:
            raise ImportError("scipy is required for robot_radius inflation (pip install scipy)")

    if invert:
        grid = np.where(grid == wall_value, free_value, wall_value).astype(np.uint8)

    info = {
        "resolution": resolution,
        "vmin_xy": vmin,     # world origin (lower-left) of grid in kept axes
        "vmax_xy": vmax,
        "H": H, "W": W,
        "up_axis": up_axis,
        "world_to_grid": world_to_grid,
        "grid_to_world": grid_to_world,
    }
    return grid, info


grid, info = mesh_to_2d_maze(
    "maze_walls.stl",
    resolution=0.02,
    pad=0.3,
    up_axis="z",
    fill=True,
    robot_radius=0.00,   # optional clearance inflation
)

# grid is your maze: 1 = wall, 0 = free
print(grid.shape, grid.dtype)