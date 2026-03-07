import os
import sys
sys.path.append(os.path.dirname(__file__))
from turtle_2d_sampler import TurtleBot2DRGBDDatasetSensorVar

sampler = TurtleBot2DRGBDDatasetSensorVar(
    dataset_path="../dataset",
    radius=0.105, #meters
    xlim=(-5, 5),
    # ylim=(-3.6, 3.6),
    ylim=(-5, 5)
)

sampler.generate_batch(
    batch_size=600,          # number of robot poses
    points_per_pose=1500
)