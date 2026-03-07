import numpy as np
import time
from pathlib import Path


class TurtleBot2DRGBDDatasetSensorVar:

    def __init__(
        self,
        dataset_path,
        radius=0.105,
        xlim=(-2, 2),
        ylim=(-2, 2),
        seed=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)

        self.r = radius
        self.xlim = xlim
        self.ylim = ylim
        self.a = 0.002
        self.b = 0.0015
        self.world_scale = 1.0

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
    # Signed distance
    # --------------------------------------------------

    def signed_distance(self, robot_xy, points):
        return np.linalg.norm(points - robot_xy, axis=1) - self.r
    
    # --------------------------------------------------
    # Sensor noise model (applied to geometry point)
    # --------------------------------------------------
    def compute_sigma(self, ranges):
        return self.a + self.b * ranges**2
    
    # --------------------------------------------------
    # Point sampling
    # --------------------------------------------------
    def sample_points_interior(self, robot_xy, num_points):
        angles = np.random.uniform(0, 2 * np.pi, num_points)

        # sqrt for uniform area sampling
        radii = self.r * np.sqrt(np.random.uniform(0.0, 1.0, num_points))

        px = robot_xy[0] + radii * np.cos(angles)
        py = robot_xy[1] + radii * np.sin(angles)

        return np.stack([px, py], axis=1)
    def sample_points_near_surface(
        self, robot_xy, num_points, offset_range=(-0.06, 0.2)
    ):
        angles = np.random.uniform(0, 2 * np.pi, num_points)
        offsets = np.random.uniform(offset_range[0], offset_range[1], num_points)
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
        batch_size=200,
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

            # -------------------------------------------------
            # TRUE DISTANCE
            # -------------------------------------------------

            gt_distance = self.signed_distance(robot_xy, points_gt)

            # -------------------------------------------------
            # RANGE SENSOR MODEL
            # -------------------------------------------------

            vec = points_gt - robot_xy
            ranges = np.linalg.norm(vec, axis=1)

            sigma = self.compute_sigma(ranges)

            noisy_ranges = ranges + np.random.normal(0.0, sigma)
            noisy_ranges = np.clip(noisy_ranges, 1e-6, None)
            # convert range → signed distance
            noisy_distance = noisy_ranges - self.r

            variance = sigma ** 2

            # -------------------------------------------------
            # FORMAT DATA SAME AS GEOMETRY DATASET
            # -------------------------------------------------

            scale = self.world_scale

            robot_scaled = robot_xy / scale
            points_scaled = points_gt / scale
            noisy_distance_scaled = noisy_distance / scale
            gt_distance_scaled = gt_distance / scale
            variance_scaled = variance / (scale**2)

            robot_repeat = np.repeat(robot_scaled[None, :], len(points_gt), axis=0)

            row = np.concatenate(
            [
                robot_repeat,
                points_scaled,
                noisy_distance_scaled[:, None],
                gt_distance_scaled[:, None],
                variance_scaled[:, None],
            ],
            axis=1,
            )

            rows.append(row)

        data = np.vstack(rows).astype(np.float32)

        ts = int(time.time())

        np.save(self.dataset_path / f"turtlebot2d_rgbd_{ts}.npy", data)
        np.save(self.dataset_path / "turtlebot2d_rgbd.npy", data)

        print("Saved dataset:", data.shape)

        return data