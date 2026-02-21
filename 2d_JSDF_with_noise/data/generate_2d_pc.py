import numpy as np
import matplotlib.pyplot as plt

WORLD_MIN, WORLD_MAX = -2.0, 2.0


# ============================================================
# WORLD
# ============================================================

def make_3_box_obstacles():
    return [
        (-1.4, -0.6, -0.2,  0.6),
        ( 0.3,  1.2, -1.2, -0.4),
        (-0.2,  0.6,  0.5,  1.4),
    ]


# ============================================================
# RGBD NOISE
# ============================================================

def add_rgbd_noise(depth):

    a = 0.002
    b = 0.0015

    sigma = a + b * depth**2
    noisy = depth + np.random.normal(0, sigma)

    noisy = np.round(noisy / 0.001) * 0.001

    dropout = np.random.rand(*depth.shape) < 0.02
    noisy[dropout] = 0.0

    noisy = np.clip(noisy, 0.0, None)

    return noisy


# ============================================================
# RAY - BOX INTERSECTION
# ============================================================

def ray_box_intersection(origin, direction, box):

    xmin, xmax, ymin, ymax = box

    tmin = (xmin - origin[0]) / direction[0] if direction[0] != 0 else -np.inf
    tmax = (xmax - origin[0]) / direction[0] if direction[0] != 0 else np.inf

    tymin = (ymin - origin[1]) / direction[1] if direction[1] != 0 else -np.inf
    tymax = (ymax - origin[1]) / direction[1] if direction[1] != 0 else np.inf

    t1 = max(min(tmin, tmax), min(tymin, tymax))
    t2 = min(max(tmin, tmax), max(tymin, tymax))

    if t2 >= max(t1, 0.0):
        return t1 if t1 > 0 else t2

    return np.inf


# ============================================================
# WORLD BOUNDARY INTERSECTION (to stop rays at edges)
# ============================================================

def ray_world_intersection(origin, direction):

    box = (WORLD_MIN, WORLD_MAX, WORLD_MIN, WORLD_MAX)
    return ray_box_intersection(origin, direction, box)


# ============================================================
# DEPTH RENDER
# ============================================================

def render_depth(pose, obstacles, fov, n_rays, max_range):

    x, y, theta = pose
    origin = np.array([x, y])

    angles = np.linspace(-fov/2, fov/2, n_rays) + theta
    depths = np.full(n_rays, max_range)

    for i, a in enumerate(angles):

        direction = np.array([np.cos(a), np.sin(a)])

        # stop at world boundary
        depths[i] = ray_world_intersection(origin, direction)

        # check obstacles
        for box in obstacles:
            d = ray_box_intersection(origin, direction, box)
            depths[i] = min(depths[i], d)

    return depths, angles


# ============================================================
# BACK PROJECT
# ============================================================

def depth_to_pointcloud(pose, depths, angles):

    x, y, _ = pose

    valid = depths > 0

    px = x + depths[valid] * np.cos(angles[valid])
    py = y + depths[valid] * np.sin(angles[valid])

    return np.stack([px, py], axis=1)


# ============================================================
# MAIN SIM
# ============================================================

def simulate_rgbd_pointcloud_2d(pose, obstacles,
                                fov=np.deg2rad(90),
                                n_rays=600):

    gt_depth, angles = render_depth(pose, obstacles, fov, n_rays, max_range=10.0)

    noisy_depth = add_rgbd_noise(gt_depth)

    pc = depth_to_pointcloud(pose, noisy_depth, angles)

    return pc


# ============================================================
# PLOT
# ============================================================

def plot_scene(pose, obstacles, pc, fov, save_path):

    x, y, theta = pose

    plt.figure(figsize=(7, 7))

    # Obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin),
                          xmax - xmin,
                          ymax - ymin,
                          fill=False,
                          linewidth=2)
        )

    # Point cloud
    plt.scatter(pc[:, 0], pc[:, 1], s=6, label="Noisy point cloud")

    # Camera
    plt.scatter(x, y, c='red', s=80, label="Camera")

    # Heading
    arrow_len = 0.3
    plt.arrow(x, y,
              arrow_len*np.cos(theta),
              arrow_len*np.sin(theta),
              head_width=0.08)

    # FOV
    for a in [-fov/2, fov/2]:
        ang = theta + a
        plt.plot([x, x + 3*np.cos(ang)],
                 [y, y + 3*np.sin(ang)],
                 linestyle="--")

    plt.xlim(WORLD_MIN, WORLD_MAX)
    plt.ylim(WORLD_MIN, WORLD_MAX)
    plt.gca().set_aspect("equal")

    plt.grid(True)
    plt.legend()
    plt.title("2D RGB-D Simulation in [-2, 2] World")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("Saved to:", save_path)

    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    obstacles = make_3_box_obstacles()

    pose = (-0.5, -0.2, np.deg2rad(120))

    pc = simulate_rgbd_pointcloud_2d(pose, obstacles)

    plot_scene(
        pose,
        obstacles,
        pc,
        fov=np.deg2rad(90),
        save_path="rgbd_sim_world.png"
    )