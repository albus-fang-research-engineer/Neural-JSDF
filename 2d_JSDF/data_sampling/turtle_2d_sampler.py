import numpy as np
import time

class TurtleBot2DSampler:

    def __init__(self, dataset_path, radius=0.105,
                 xlim=(-2, 2), ylim=(-2, 2)):
        self.dataset_path = dataset_path
        self.r = radius
        self.xlim = xlim
        self.ylim = ylim

    def sample_robot_pose(self):
        x = np.random.uniform(*self.xlim)
        y = np.random.uniform(*self.ylim)
        return np.array([x, y])

    def signed_distance(self, robot_xy, points):
        diff = points - robot_xy
        dist = np.linalg.norm(diff, axis=1)
        return dist - self.r   # outside positive

    def sample_points_near_surface(self, robot_xy, num_points,
                                   offset_range=(-0.03, 0.2)):

        angles = np.random.uniform(0, 2*np.pi, num_points)

        offsets = np.random.uniform(offset_range[0],
                                    offset_range[1],
                                    num_points)

        radii = self.r + offsets

        px = robot_xy[0] + radii * np.cos(angles)
        py = robot_xy[1] + radii * np.sin(angles)

        return np.stack([px, py], axis=1)

    def sample_points_uniform(self, num_points):
        px = np.random.uniform(*self.xlim, num_points)
        py = np.random.uniform(*self.ylim, num_points)
        return np.stack([px, py], axis=1)

    def generate_batch(self,
                       batch_size=10000,
                       near_surface_ratio=0.7,
                       points_per_pose=10000):

        data = []

        for _ in range(batch_size):

            robot_xy = self.sample_robot_pose()

            n_surface = int(points_per_pose * near_surface_ratio)
            n_uniform = points_per_pose - n_surface

            pts_surface = self.sample_points_near_surface(
                robot_xy, n_surface)

            pts_uniform = self.sample_points_uniform(n_uniform)

            points = np.vstack([pts_surface, pts_uniform])

            d = self.signed_distance(robot_xy, points)

            robot_repeat = np.repeat(robot_xy[None, :],
                                     len(points),
                                     axis=0)

            row = np.concatenate([robot_repeat,
                                  points,
                                  d[:, None]], axis=1)

            data.append(row)

        data = np.vstack(data).astype(np.float32)

        np.save(f"{self.dataset_path}/turtlebot2d_{int(time.time())}.npy",
                data)
        np.save(f"{self.dataset_path}/turtlebot2d.npy",
                data)
        print("Saved dataset:", data.shape)