import os
import sys
sys.path.append(os.path.dirname(__file__))
from turtle_2d_sampler_noisy_data import TurtleBot2DGeometryDataset

sampler = TurtleBot2DGeometryDataset(
    dataset_path="../dataset",
    radius=0.0105,
    xlim=(-0.5, 0.5),
    ylim=(-0.36, 0.36)
)

sampler.generate_batch(
    batch_size=600,          # number of robot poses
    points_per_pose=1500
)