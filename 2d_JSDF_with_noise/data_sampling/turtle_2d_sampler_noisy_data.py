import numpy as np
import time
from pathlib import Path


class TurtleBot2DNoisyGeometryDataset:

    def __init__(
        self,
        dataset_path,
        radius=0.105,
        xlim=(-2, 2),
        ylim=(-2, 2),
        noise_a=0.002,
        noise_b=0.0015,
        seed=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)

        self.r = radius
        self.xlim = xlim
        self.ylim = ylim

        self.a = noise_a
        self.b = noise_b

        if seed is not None:
            np.random.seed(seed)

    # --------------------------------------------------
    # Robot pose
    # --------------------------------------------------

    def sample_robot_pose(self):
        x = np.random.uniform(*self.xlim)
        y = np.random.uniform(*self.ylim)
        return np.array([x, y])

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    def signed_distance(self, robot_xy, points):
        diff = points - robot_xy
        dist = np.linalg.norm(diff, axis=1)
        return dist - self.r

    # --------------------------------------------------
    # Sensor noise model (applied to geometry point)
    # --------------------------------------------------
    def compute_sigma(self, ranges):
        return self.a + self.b * ranges**2


    def add_range_noise(self, robot_xy, points_gt):
        vec = points_gt - robot_xy
        ranges = np.linalg.norm(vec, axis=1)
        ranges = np.clip(ranges, 1e-6, None)
        # unit ray directions
        dirs = vec / ranges[:, None]

        sigma = self.compute_sigma(ranges)

        noisy_ranges = ranges + np.random.normal(0.0, sigma)

        points_noisy = robot_xy + dirs * noisy_ranges[:, None]

        return points_noisy, sigma
    # def compute_sigma(self, robot_xy, points):
    #     ranges = np.linalg.norm(points - robot_xy, axis=1)
    #     return self.a + self.b * ranges**2

    # def add_point_noise(self, points, sigma):
    #     noise = np.random.normal(0.0, sigma[:, None], size=points.shape)
    #     return points + noise

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    def sample_points_near_surface(
        self, robot_xy, num_points, offset_range=(-0.03, 0.2)
    ):
        angles = np.random.uniform(0, 2 * np.pi, num_points)

        offsets = np.random.uniform(
            offset_range[0], offset_range[1], num_points
        )

        radii = self.r + offsets

        px = robot_xy[0] + radii * np.cos(angles)
        py = robot_xy[1] + radii * np.sin(angles)

        return np.stack([px, py], axis=1)

    def sample_points_uniform(self, num_points):
        px = np.random.uniform(*self.xlim, num_points)
        py = np.random.uniform(*self.ylim, num_points)
        return np.stack([px, py], axis=1)

    # --------------------------------------------------
    # Dataset generation
    # --------------------------------------------------

    def generate_batch(
        self,
        batch_size=100,
        points_per_pose=500,
        near_surface_ratio=0.6,
    ):

        rows = []

        for _ in range(batch_size):

            robot_xy = self.sample_robot_pose()

            n_surface = int(points_per_pose * near_surface_ratio)
            n_uniform = points_per_pose - n_surface

            pts_surface = self.sample_points_near_surface(robot_xy, n_surface)
            pts_uniform = self.sample_points_uniform(n_uniform)

            points_gt = np.vstack([pts_surface, pts_uniform])

            gt_distance = self.signed_distance(robot_xy, points_gt)

            # sigma = self.compute_sigma(robot_xy, points_gt)
            # var = sigma**2

            # points_noisy = self.add_point_noise(points_gt, sigma)
            ranges = np.linalg.norm(points_gt - robot_xy, axis=1)

            points_noisy, sigma = self.add_range_noise(robot_xy, points_gt)
            var = sigma**2

            noisy_distance = self.signed_distance(robot_xy, points_noisy)

            robot_repeat = np.repeat(robot_xy[None, :], len(points_gt), axis=0)

            row = np.concatenate(
                [
                    robot_repeat,
                    points_noisy,
                    noisy_distance[:, None],
                    gt_distance[:, None],
                    var[:, None],
                ],
                axis=1,
            )

            rows.append(row)

        data = np.vstack(rows).astype(np.float32)

        ts = int(time.time())

        np.save(self.dataset_path / f"turtlebot2d_geom_{ts}.npy", data)
        np.save(self.dataset_path / "turtlebot2d_geom.npy", data)

        print("Saved dataset:", data.shape)

        return data


if __name__ == "__main__":

    ds = TurtleBot2DNoisyGeometryDataset(
        dataset_path="dataset_out",
        noise_a=0.002,
        noise_b=0.0015,
        seed=0,
    )

    ds.generate_batch(
        batch_size=200,
        points_per_pose=512,
        near_surface_ratio=0.7,
    )