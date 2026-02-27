import os
import sys
sys.path.append(os.path.dirname(__file__))
from turtle_2d_sampler_noisy_data import TurtleBot2DGeometryDataset

sampler = TurtleBot2DGeometryDataset(
    dataset_path="../dataset",
    radius=0.105,
    xlim=(-2, 2),
    ylim=(-2, 2)
)

sampler.generate_batch(
    batch_size=3000,          # number of robot poses
    points_per_pose=1500
)