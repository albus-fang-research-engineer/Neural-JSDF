import numpy as np
import matplotlib.pyplot as plt

ROBOT_RADIUS = 0.105  # meters
N = 500             # points per pose (must match your generator)


def plot_samples(data, pose_id, title_prefix="TurtleBot 2D"):

    robot_xy = data[0, 0:2]
    points = data[:, 2:4]
    dist = data[:, 4]

    inside = dist < 0
    outside = dist >= 0

    plt.figure(figsize=(6, 6))

    plt.scatter(points[outside, 0],
                points[outside, 1],
                s=8,
                c='red',
                label="outside")

    plt.scatter(points[inside, 0],
                points[inside, 1],
                s=8,
                c='blue',
                label="inside")

    circle = plt.Circle(robot_xy,
                        ROBOT_RADIUS,
                        fill=False,
                        linewidth=2)

    plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    title = f"{title_prefix} — pose {pose_id}"
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.savefig(f"plots/sampled_points/plot_pose_{pose_id}.png")
    plt.close()


def plot_multiple_poses(data, num_poses_to_plot=5, random=True):
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

        plot_samples(pose_block, pose_id)


if __name__ == "__main__":

    data = np.load("../dataset/turtlebot2d.npy")

    plot_multiple_poses(
        data,
        num_poses_to_plot=6,   # ← change this
        random=True            # ← False = first K poses in order
    )