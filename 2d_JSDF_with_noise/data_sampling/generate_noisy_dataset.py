import os
import sys
sys.path.append(os.path.dirname(__file__))
from turtle_2d_sampler_lidar import TurtleBot2DLidarDatasetSensorVar

sampler = TurtleBot2DLidarDatasetSensorVar(
    dataset_path="../dataset",
    radius=0.105, #meters
    xlim=(-5, 5),
    ylim=(-3.6, 3.6),
    # ylim=(-5, 5)
)

sampler.generate_batch(
    batch_size=1000,          # number of robot poses
    points_per_pose=1000
)