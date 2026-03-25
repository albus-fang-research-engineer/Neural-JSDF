import torch
import numpy as np
from sdf.robot_sdf import RobotSdfCollisionNet


class NJSDFHelper:
    def __init__(self, model_path, num_joints=6, device="cpu"):
        """
        model_path: path to .pt weights
        num_joints: e.g. 6 for UR5e
        """
        self.device = torch.device(device)
        self.num_joints = num_joints

        # Build model (same config as training)
        s = 256
        n_layers = 5
        skips = []

        if skips == []:
            n_layers -= 1

        # input = q + xyz
        in_channels = num_joints + 3

        # NOTE: out_channels will be inferred after loading
        self.nn_model = RobotSdfCollisionNet(
            in_channels=in_channels,
            out_channels=7,
            layers=[s] * n_layers,
            skips=skips
        )

        # Load weights
        self.nn_model.load_weights(model_path, {'device': self.device, 'dtype': torch.float32})
        self.model = self.nn_model.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, q, points):
        """
        q: (6,) or (N,6)
        points: (3,) or (N,3)

        Returns:
            distances: (N, num_links) or (num_links,)
        """

        q = np.atleast_2d(q)
        points = np.atleast_2d(points)

        assert q.shape[1] == self.num_joints
        assert points.shape[1] == 3

        # broadcast if needed
        if q.shape[0] == 1 and points.shape[0] > 1:
            q = np.repeat(q, points.shape[0], axis=0)
        if points.shape[0] == 1 and q.shape[0] > 1:
            points = np.repeat(points, q.shape[0], axis=0)

        x = np.concatenate([q, points], axis=1)

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

        y_pred = self.model(x_tensor)

        # IMPORTANT: undo scaling (they trained with *100)
        y_pred = y_pred / 100.0

        return y_pred.cpu().numpy()