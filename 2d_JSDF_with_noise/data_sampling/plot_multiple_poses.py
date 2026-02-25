import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROBOT_RADIUS = 0.105


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def infer_points_per_pose(data):
    robot_xy = data[:, 0:2]
    first_pose = robot_xy[0]
    change_idx = np.where(np.linalg.norm(robot_xy - first_pose, axis=1) > 1e-6)[0]
    return change_idx[0] if len(change_idx) > 0 else len(data)


def subsample_indices(n, max_points=200):
    step = max(1, n // max_points)
    return np.arange(0, n, step)


# ------------------------------------------------------------
# Plot error heatmap
# ------------------------------------------------------------

def plot_pose_error(pose_block, pose_id, outdir):

    robot_xy = pose_block[0, 0:2]
    points = pose_block[:, 2:4]

    noisy_d = pose_block[:, 4]
    gt_d = pose_block[:, 5]

    error = np.abs(noisy_d - gt_d)

    idx = subsample_indices(len(points))

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(points[idx, 0],
                     points[idx, 1],
                     c=error[idx],
                     s=10,
                     cmap="inferno")

    plt.colorbar(sc, label="|noisy − GT distance| [m]")

    circle = plt.Circle(robot_xy,
                        ROBOT_RADIUS,
                        fill=False,
                        linewidth=2)

    plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Pose {pose_id} — distance error heatmap")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"pose_{pose_id}_error.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# Plot variance heatmap
# ------------------------------------------------------------

def plot_pose_variance(pose_block, pose_id, outdir):

    robot_xy = pose_block[0, 0:2]
    points = pose_block[:, 2:4]

    var = pose_block[:, 6]

    idx = subsample_indices(len(points))

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(points[idx, 0],
                     points[idx, 1],
                     c=var[idx],
                     s=10,
                     cmap="viridis")

    plt.colorbar(sc, label="GT sensor variance")

    circle = plt.Circle(robot_xy,
                        ROBOT_RADIUS,
                        fill=False,
                        linewidth=2)

    plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Pose {pose_id} — aleatoric variance")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"pose_{pose_id}_variance.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def plot_multiple_poses(data, num_poses_to_plot=5, random=True):

    N = infer_points_per_pose(data)
    total_poses = data.shape[0] // N

    if random:
        pose_ids = np.random.choice(total_poses,
                                    size=num_poses_to_plot,
                                    replace=False)
    else:
        pose_ids = np.arange(min(num_poses_to_plot, total_poses))

    for pose_id in pose_ids:

        start = pose_id * N
        end = start + N
        pose_block = data[start:end]

        plot_pose_error(
            pose_block,
            pose_id,
            Path("plots/error")
        )

        plot_pose_variance(
            pose_block,
            pose_id,
            Path("plots/variance")
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    data = np.load("../dataset/turtlebot2d_geom.npy")

    plot_multiple_poses(
        data,
        num_poses_to_plot=6,
        random=True
    )