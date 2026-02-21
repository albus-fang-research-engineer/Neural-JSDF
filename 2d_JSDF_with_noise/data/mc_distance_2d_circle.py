import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

WORLD_MIN, WORLD_MAX = -2.0, 2.0
ROBOT_RADIUS = 0.105


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
# NOISE MODEL
# ============================================================

def add_rgbd_noise(depth):

    a = 0.002
    b = 0.0015

    sigma = a + b * depth**2
    noisy = depth + np.random.normal(0, sigma)

    noisy = np.round(noisy / 0.001) * 0.001

    dropout = np.random.rand(*depth.shape) < 0.02
    noisy[dropout] = np.nan   # IMPORTANT: mark invalid

    return noisy


# ============================================================
# RAY INTERSECTION
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


def ray_world_intersection(origin, direction):
    world_box = (WORLD_MIN, WORLD_MAX, WORLD_MIN, WORLD_MAX)
    return ray_box_intersection(origin, direction, world_box)


# ============================================================
# DEPTH RENDER
# ============================================================

def render_depth(pose, obstacles, fov, n_rays):

    x, y, theta = pose
    origin = np.array([x, y])

    angles = np.linspace(-fov/2, fov/2, n_rays) + theta
    depths = np.zeros(n_rays)

    for i, a in enumerate(angles):

        direction = np.array([np.cos(a), np.sin(a)])

        d = ray_world_intersection(origin, direction)

        for box in obstacles:
            d = min(d, ray_box_intersection(origin, direction, box))

        depths[i] = d

    return depths, angles


# ============================================================
# DISTANCE TO CIRCLE ROBOT
# ============================================================

def point_to_circle_distance(points, robot_xy):

    d = np.linalg.norm(points - robot_xy, axis=1) - ROBOT_RADIUS
    return np.maximum(d, 0.0)


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_ray_statistics(
    pose,
    obstacles,
    n_rays=640,
    fov=np.deg2rad(90),
    n_mc=300,
):

    robot_xy = np.array(pose[:2])

    gt_depth, angles = render_depth(pose, obstacles, fov, n_rays)

    dist_samples = np.full((n_mc, n_rays), np.nan)

    for k in range(n_mc):

        noisy_depth = add_rgbd_noise(gt_depth)

        valid = ~np.isnan(noisy_depth)

        px = robot_xy[0] + noisy_depth[valid] * np.cos(angles[valid])
        py = robot_xy[1] + noisy_depth[valid] * np.sin(angles[valid])

        points = np.stack([px, py], axis=1)

        dist = point_to_circle_distance(points, robot_xy)

        dist_samples[k, valid] = dist

    mean_dist = np.nanmean(dist_samples, axis=0)
    var_dist = np.nanvar(dist_samples, axis=0)
    valid_rate = np.mean(~np.isnan(dist_samples), axis=0)

    return angles, mean_dist, var_dist, valid_rate, gt_depth


# ============================================================
# SAVE DATASET
# ============================================================

def save_npz(output_path, pose, angles, mean_dist, var_dist, valid_rate):

    np.savez_compressed(
        output_path,
        robot_pose=np.array(pose),
        angles=angles,
        mean_dist=mean_dist,
        var_dist=var_dist,
        valid_rate=valid_rate,
    )


# ============================================================
# DEBUG PLOT
# ============================================================

def plot_statistics(angles, mean_dist, var_dist, save_path):

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(angles, mean_dist)
    axs[0].set_title("Mean distance per ray")

    axs[1].plot(angles, var_dist)
    axs[1].set_title("Variance per ray")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def compute_gt_distances(pose, gt_depth, angles):

    robot_xy = np.array(pose[:2])

    px = robot_xy[0] + gt_depth * np.cos(angles)
    py = robot_xy[1] + gt_depth * np.sin(angles)

    points = np.stack([px, py], axis=1)

    gt_dist = np.linalg.norm(points - robot_xy, axis=1) - ROBOT_RADIUS
    gt_dist = np.maximum(gt_dist, 0.0)

    return points, gt_dist


def plot_world_points_with_stats(
    pose,
    obstacles,
    points,          # (N,2) GT hit points in world
    gt_dist,         # (N,)
    mean_dist,       # (N,)
    var_dist,        # (N,)
    fov,
    save_path,
    annotate_every=40,        # label every k-th point
    annotate_topk_var=0,      # additionally label top-k highest variance points (0 disables)
    point_size=10
):
    x, y, theta = pose

    plt.figure(figsize=(8, 8))

    # Obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          fill=False, linewidth=2)
        )

    # Points colored by variance
    sc = plt.scatter(points[:, 0], points[:, 1], c=var_dist, s=point_size)
    plt.colorbar(sc, label="Variance of distance (m^2)")

    # Robot/camera circle + pose marker
    plt.gca().add_patch(plt.Circle((x, y), ROBOT_RADIUS, fill=False, linewidth=2))
    plt.scatter([x], [y], s=80, marker="x", label="Camera/Robot center")

    # Heading arrow
    arrow_len = 0.35
    plt.arrow(
        x, y,
        arrow_len * np.cos(theta),
        arrow_len * np.sin(theta),
        head_width=0.08,
        length_includes_head=True
    )

    # View cone (FOV lines)
    cone_len = 2.5
    for a in (-fov / 2, fov / 2):
        ang = theta + a
        plt.plot(
            [x, x + cone_len * np.cos(ang)],
            [y, y + cone_len * np.sin(ang)],
            linestyle="--",
            linewidth=1.5
        )

    # Decide which points to annotate
    N = points.shape[0]
    idx = set(range(0, N, max(1, annotate_every)))

    if annotate_topk_var and annotate_topk_var > 0:
        topk = np.argsort(var_dist)[-annotate_topk_var:]
        idx.update(topk.tolist())

    idx = sorted(idx)

    # Annotate selected points
    for i in idx:
        px, py = points[i]
        text = f"gt={gt_dist[i]:.3f}\nμ={mean_dist[i]:.3f}\nσ²={var_dist[i]:.2e}"
        plt.text(px, py, text, fontsize=7)

    # Fixed world bounds
    plt.xlim(WORLD_MIN, WORLD_MAX)
    plt.ylim(WORLD_MIN, WORLD_MAX)
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.title("World hits colored by variance; annotations show gt / mean / var")
    plt.legend(loc="upper right")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
def select_plot_indices(var_dist, max_points=60, include_topk=10):
    N = len(var_dist)

    stride = max(1, N // max_points)
    base_idx = np.arange(0, N, stride)

    if include_topk > 0:
        topk = np.argsort(var_dist)[-include_topk:]
        idx = np.unique(np.concatenate([base_idx, topk]))
    else:
        idx = base_idx

    return idx
def gt_hit_points_and_gt_dist(pose, gt_depth, angles):
    x, y, _ = pose
    robot_xy = np.array([x, y])

    px = x + gt_depth * np.cos(angles)
    py = y + gt_depth * np.sin(angles)
    points = np.stack([px, py], axis=1)

    gt_dist = np.maximum(np.linalg.norm(points - robot_xy, axis=1) - ROBOT_RADIUS, 0.0)
    return points, gt_dist
def spatially_filter_indices(points, candidate_idx, min_dist=0.15):
    selected = []
    for i in candidate_idx:
        p = points[i]

        if all(np.linalg.norm(p - points[j]) > min_dist for j in selected):
            selected.append(i)

    return np.array(selected)
# ============================================================
# RUN EXAMPLE
# ============================================================

if __name__ == "__main__":

    output_dir = Path("mc_dataset")
    output_dir.mkdir(exist_ok=True)

    pose = (-1.0, -1.0, np.deg2rad(30))

    obstacles = make_3_box_obstacles()

    angles, mean_dist, var_dist, valid_rate, gt_depth = monte_carlo_ray_statistics(
        pose,
        obstacles,
        n_rays=640,
        n_mc=400,
    )
    points, gt_dist = compute_gt_distances(pose, gt_depth, angles)
    save_npz(
        output_dir / "pose_000.npz",
        pose,
        angles,
        mean_dist,
        var_dist,
        valid_rate,
    )

    plot_statistics(
        angles,
        mean_dist,
        var_dist,
        output_dir / "pose_000_stats.png"
    )

    obstacles = make_3_box_obstacles()
    fov = np.deg2rad(90)

    angles, mean_dist, var_dist, valid_rate, gt_depth = monte_carlo_ray_statistics(
        pose, obstacles, n_rays=640, fov=fov, n_mc=400
    )

    points, gt_dist = gt_hit_points_and_gt_dist(pose, gt_depth, angles)
    idx = select_plot_indices(var_dist, max_points=30, include_topk=10)
    idx = spatially_filter_indices(points, idx, min_dist=0.18)
    points_plot = points[idx]
    gt_plot     = gt_dist[idx]
    mean_plot   = mean_dist[idx]
    var_plot    = var_dist[idx]
    plot_world_points_with_stats(
        pose=pose,
        obstacles=obstacles,
        points=points_plot,
        gt_dist=gt_plot,
        mean_dist=mean_plot,
        var_dist=var_plot,
        fov=fov,
        save_path="mc_dataset/pose_000_world_stats.png",
        annotate_every=50,        # tune this
        annotate_topk_var=10      # label the 10 spikiest rays too
    )