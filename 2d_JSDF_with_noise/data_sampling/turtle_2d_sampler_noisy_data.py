import numpy as np
import time
from pathlib import Path


class TurtleBot2DGeometryDataset:

    def __init__(
        self,
        dataset_path,
        radius=0.105,
        xlim=(-2, 2),
        ylim=(-2, 2),
        dist_sigma=0.02,
        seed=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.mkdir(exist_ok=True)

        self.r = radius
        self.xlim = xlim
        self.ylim = ylim
        self.sigma = dist_sigma

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
        var = self.sigma ** 2  # constant variance

        for _ in range(batch_size):

            robot_xy = self.sample_robot_pose()

            n_surface = int(points_per_pose * near_surface_ratio)
            n_uniform = points_per_pose - n_surface

            pts_surface = self.sample_points_near_surface(robot_xy, n_surface)
            pts_uniform = self.sample_points_uniform(n_uniform)

            points_gt = np.vstack([pts_surface, pts_uniform])

            gt_distance = self.signed_distance(robot_xy, points_gt)

            noisy_distance = gt_distance + np.random.normal(
                0.0, self.sigma, size=gt_distance.shape
            )

            robot_repeat = np.repeat(robot_xy[None, :], len(points_gt), axis=0)

            row = np.concatenate(
                [
                    robot_repeat,                 # robot_x, robot_y
                    points_gt,                   # point_x, point_y (GT!)
                    noisy_distance[:, None],    # noisy distance
                    gt_distance[:, None],       # GT distance
                    np.full((len(points_gt), 1), var),  # constant variance
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