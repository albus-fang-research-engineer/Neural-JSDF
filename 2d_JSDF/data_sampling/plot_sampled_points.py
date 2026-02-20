import numpy as np
import matplotlib.pyplot as plt


ROBOT_RADIUS = 0.105  # meters


def plot_samples(data, title="Sampled Points"):

    # Split columns
    robot_xy = data[0, 0:2]        # same for all rows in a batch
    points = data[:, 2:4]
    dist = data[:, 4]

    inside = dist < 0
    outside = dist >= 0

    plt.figure(figsize=(6, 6))

    # outside points
    plt.scatter(points[outside, 0],
                points[outside, 1],
                s=8,
                c='red',
                label="outside")

    # inside points
    plt.scatter(points[inside, 0],
                points[inside, 1],
                s=8,
                c='blue',
                label="inside")

    # robot footprint
    circle = plt.Circle(robot_xy,
                        ROBOT_RADIUS,
                        fill=False,
                        linewidth=2)

    plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.savefig("plot.png")
    # plt.show()


if __name__ == "__main__":

    
    data = np.load("../dataset/turtlebot2d.npy")

    # If this file contains multiple robot poses,
    # pick one pose block to visualize:
    N = 10000  # points per pose (must match your generator)
    sample_block = data[:N]

    plot_samples(sample_block, title="TurtleBot 2D Sampling")