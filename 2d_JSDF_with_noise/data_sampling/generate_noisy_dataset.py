import os
import sys
sys.path.append(os.path.dirname(__file__))
from turtle_2d_sampler_noisy_data import TurtleBot2DNoisyGeometryDataset

sampler = TurtleBot2DNoisyGeometryDataset(
    dataset_path="../dataset",
    radius=0.105,
    xlim=(-2, 2),
    ylim=(-2, 2)
)

sampler.generate_batch(
    batch_size=400,          # number of robot poses
    points_per_pose=500
)