import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sdf.stochastic_robot_sdf import RobotSdfCollisionNet
from obstacle_eval_utils import (
    make_3_box_obstacles,
    point_in_any_box,
    sample_points_avoiding_boxes,
    sample_box_surface,
    demo_obstacles_and_points,
    demo_obstacles_and_20_points,
    dense_obstacle_evaluation,
    demo_visible_obstacles_and_points
)
scale_factor = 10.0
# MODEL_PATH = "sdf_2d_rgbd_truepoint_noisylabel.pt"
MODEL_PATH = "sdf_2d_lidar_mesh_units.pt"
RADIUS = 0.105 / scale_factor
RADIUS_PLOT_FACTOR = 2
def predict_mu_var(model, x_tensor, logvar_min=-20.0, logvar_max=10.0):
    """
    model(x) -> [mu, logvar] concatenated along last dim (shape [B,2] for out_channels=1)
    returns: mu [B], var [B]
    """
    pred = model(x_tensor)
    mu, logvar = torch.chunk(pred, 2, dim=-1)
    logvar = torch.clamp(logvar, logvar_min, logvar_max)
    var = torch.exp(logvar)
    return mu.squeeze(-1), var.squeeze(-1)
# -------------------------------------------------
# ANALYTIC GROUND TRUTH (circle SDF)
# -------------------------------------------------
def gt_signed_distance(robot_xy, points):
    diff = points - robot_xy
    return np.linalg.norm(diff, axis=1) - RADIUS

def gt_distance_sigma(robot_xy, points, scale_factor=10.0):
    # ranges in mesh units
    ranges_mesh = np.linalg.norm(points - robot_xy, axis=1)

    # convert to meters for physical noise model
    ranges_m = ranges_mesh * scale_factor

    sigma_m = np.empty_like(ranges_m)

    # --- masks in METERS ---
    mask1 = ranges_m < 0.8
    mask2 = (ranges_m >= 0.8) & (ranges_m < 1.8)
    mask3 = (ranges_m >= 1.8) & (ranges_m < 2.6)
    mask4 = ranges_m >= 2.6

    # --- Segment 1 ---
    sigma_m[mask1] = 0.01

    # --- Segment 2 ---
    sigma_m[mask2] = 0.01 + 0.01 * (ranges_m[mask2] - 0.8)

    # --- Segment 3 (quadratic bump) ---
    d3 = ranges_m[mask3] - 1.8
    sigma_m[mask3] = (
        0.02
        + 0.01 * d3
        + 0.036 * d3**2
    )

    # --- Anchor at 2.6 ---
    d_anchor = 2.6 - 1.8
    sigma_anchor = (
        0.02
        + 0.01 * d_anchor
        + 0.036 * d_anchor**2
    )

    # --- Segment 4 (slow linear growth) ---
    sigma_m[mask4] = sigma_anchor + 0.006 * (ranges_m[mask4] - 2.6)

    # convert back to mesh units
    sigma_mesh = sigma_m / scale_factor
    return sigma_mesh

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobotSdfCollisionNet(
    in_channels=4,
    out_channels=1,
    layers=[128] * 4,
    skips=[]
).model

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

print("Model loaded")


# -------------------------------------------------
# GENERATE TEST DATA
# -------------------------------------------------
N = 20000

robot_xy = np.array([0.0, 0.0])  # fixed robot for visualization

px = np.random.uniform(-5, 5, N) /scale_factor
py = np.random.uniform(-5, 5, N) / scale_factor

points = np.stack([px, py], axis=1)

gt = gt_signed_distance(robot_xy, points)

x_input = np.concatenate(
    [np.repeat(robot_xy[None, :], N, axis=0), points],
    axis=1
)

x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)


# -------------------------------------------------
# INFERENCE SPEED TEST
# -------------------------------------------------
torch.cuda.synchronize() if device.type == "cuda" else None
t0 = time.time()

# with torch.no_grad():
#     pred = model(x_tensor).squeeze().cpu().numpy()
with torch.no_grad():
    mu_t, var_t = predict_mu_var(model, x_tensor)
    pred = mu_t.cpu().numpy()
    pred_var = var_t.cpu().numpy()

torch.cuda.synchronize() if device.type == "cuda" else None
t1 = time.time()

dt = t1 - t0

print(f" Inference time for {N} samples: {dt:.6f} s")
print(f" Samples/sec: {N/dt:.2f}")


# -------------------------------------------------
# ERROR METRICS
# -------------------------------------------------
mae = np.mean(np.abs(pred - gt))
rmse = np.sqrt(np.mean((pred - gt) ** 2))

print("MAE:", mae)
print("RMSE:", rmse)



NUM_POSES = 4
POINTS_PER_POSE = 4000

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
all_var = []
all_err = []
for ax in axes.flat:

    # ----------------------------
    # random robot pose
    # ----------------------------
    robot_xy = np.random.uniform(-4.5, 4.5, size=2) / scale_factor

    px = np.random.uniform(-5, 5, POINTS_PER_POSE) / scale_factor
    py = np.random.uniform(-5, 5, POINTS_PER_POSE) / scale_factor
    points = np.stack([px, py], axis=1)

    gt = gt_signed_distance(robot_xy, points)
    gt_sigma = gt_distance_sigma(robot_xy, points)
    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], POINTS_PER_POSE, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        x_t = torch.tensor(x_input, dtype=torch.float32, device=device)
        pred_out = model(x_t)

        mu, logvar = torch.chunk(pred_out, 2, dim=-1)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        var = torch.exp(logvar)

        pred = mu.cpu().numpy().squeeze()
        pred_var = var.cpu().numpy().squeeze()

    error = np.abs(pred - gt)
    all_var.append(pred_var)
    all_err.append(error)
    # ----------------------------
    # plot points colored by error
    # ----------------------------
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=error,
        cmap="coolwarm",
        s=3
    )

    # robot footprint
    circle = plt.Circle(robot_xy, RADIUS * RADIUS_PLOT_FACTOR,color="red",edgecolor="black",linewidth=1.5,alpha=0.9,zorder=5)
    ax.add_patch(circle)

    ax.set_title(f"Robot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)

fig.colorbar(sc, ax=axes.ravel().tolist(), location="right", label="|Prediction Error| [m]")
# plt.tight_layout()
plt.savefig("stochastic_plots/spatial_error.png", dpi=200)


all_var = np.concatenate(all_var)
all_err = np.concatenate(all_err)

plt.figure(figsize=(6, 5))
plt.scatter(all_var, all_err, s=2, alpha=0.3)
plt.xlabel("Predicted variance σ²")
plt.ylabel("|μ - GT|")
plt.title("Uncertainty vs absolute error")
plt.grid(True)
plt.tight_layout()
plt.savefig("stochastic_plots/var_vs_abs_error.png", dpi=200)
print("Saved: var_vs_abs_error.png")
# -------------------------------------------------
# SPATIAL VARIANCE PLOT (same layout)
# -------------------------------------------------
fig_var, axes_var = plt.subplots(2, 2, figsize=(10, 10))

for ax in axes_var.flat:

    robot_xy = np.random.uniform(-4.5, 4.5, size=2) / scale_factor

    px = np.random.uniform(-5, 5, POINTS_PER_POSE) / scale_factor
    py = np.random.uniform(-5, 5, POINTS_PER_POSE) / scale_factor
    points = np.stack([px, py], axis=1)

    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], POINTS_PER_POSE, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        x_t = torch.tensor(x_input, dtype=torch.float32, device=device)
        pred_out = model(x_t)

        mu, logvar = torch.chunk(pred_out, 2, dim=-1)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        var = torch.exp(logvar)

        pred_var = var.cpu().numpy().squeeze()

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=pred_var,
        cmap="viridis",
        s=3
    )

    circle = plt.Circle(robot_xy, RADIUS*RADIUS_PLOT_FACTOR,color="red",edgecolor="black",linewidth=1.5,alpha=0.9,zorder=5)
    ax.add_patch(circle)

    ax.set_title(f"Predicted σ² @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)

fig_var.colorbar(sc, ax=axes_var.ravel().tolist(), location="right", label="Predicted variance σ²")
plt.savefig("stochastic_plots/spatial_variance.png", dpi=200)

print("Saved: spatial_variance.png")

# -------------------------------------------------
# QUALITATIVE CHECK: 5 poses, 1 point each
# -------------------------------------------------
NUM_TEST = 5

fig, axes = plt.subplots(1, NUM_TEST, figsize=(4 * NUM_TEST, 4))

if NUM_TEST == 1:
    axes = [axes]

for ax in axes:

    # random robot pose
    robot_xy = np.random.uniform(-4.5, 4.5, size=2) / scale_factor

    # one random query point
    point = np.random.uniform(-5, 5, size=(1, 2)) / scale_factor

    # ground truth
    gt = gt_signed_distance(robot_xy, point)[0]
    gt_sigma = gt_distance_sigma(robot_xy, point)[0]
    print("gt_variance:", gt_sigma**2)
    # model input
    x_input = np.concatenate(
        [robot_xy[None, :], point],
        axis=1
    )

    with torch.no_grad():
        x_t = torch.tensor(x_input, dtype=torch.float32, device=device)
        pred_out = model(x_t)

        mu, logvar = torch.chunk(pred_out, 2, dim=-1)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        var = torch.exp(logvar)

        pred = float(mu.item())
        pred_var = float(var.item())

    # ----------------------------
    # PLOT
    # ----------------------------
    ax.scatter(point[0, 0], point[0, 1], s=80)

    circle = plt.Circle(robot_xy, RADIUS*RADIUS_PLOT_FACTOR,color="red",edgecolor="black",linewidth=1.5,alpha=0.9,zorder=5)
    ax.add_patch(circle)

    ax.text(
        point[0, 0],
        point[0, 1],
        # f"GT: {gt:.3f}\nPred: {pred:.3f}",
        f"μGT: {gt:.3f}\nμ: {pred:.3f}\nσ: {np.sqrt(pred_var):.6f}\nσGT: {gt_sigma:.6f}",
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.set_title(f"Robot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)

plt.tight_layout()

plt.savefig("stochastic_plots/five_random_pose_single_point.png", dpi=200)
print("Saved: five_random_pose_single_point.png")

# -------------------------------------------------
# QUALITATIVE CHECK FROM DATASET
# -------------------------------------------------
NUM_TEST = 5

data = np.load("../dataset/turtlebot2d_lidar.npy")

fig, axes = plt.subplots(1, NUM_TEST, figsize=(4 * NUM_TEST, 4))
if NUM_TEST == 1:
    axes = [axes]

num_samples = data.shape[0]

for ax in axes:

    # ----------------------------
    # random row from dataset
    # ----------------------------
    idx = np.random.randint(0, num_samples)

    robot_xy = data[idx, 0:2]
    point    = data[idx, 2:4]

    gt_mu  = data[idx, 5]
    gt_var = data[idx, 6]

    # model input
    x_input = data[idx, 0:4][None, :]

    # ----------------------------
    # MODEL INFERENCE
    # ----------------------------
    with torch.no_grad():
        x_t = torch.tensor(x_input, dtype=torch.float32, device=device)

        pred_out = model(x_t)

        mu, logvar = torch.chunk(pred_out, 2, dim=-1)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        var = torch.exp(logvar)

        pred_mu  = float(mu.item())
        pred_var = float(var.item())

    # ----------------------------
    # PLOT
    # ----------------------------
    ax.scatter(point[0], point[1], s=80)

    circle = plt.Circle(robot_xy, RADIUS*RADIUS_PLOT_FACTOR,color="red",edgecolor="black",linewidth=1.5,alpha=0.9,zorder=5)
    ax.add_patch(circle)

    ax.text(
        point[0],
        point[1],
        (
            f"μGT: {gt_mu:.3f}\n"
            f"μ: {pred_mu:.3f}\n"
            f"σGT: {np.sqrt(gt_var):.6f}\n"
            f"σ: {np.sqrt(pred_var):.6f}"
        ),
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.set_title(f"Robot @ {robot_xy.round(5)}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(True)

plt.tight_layout()
plt.savefig("stochastic_plots/five_training_pose_single_point_dataset.png", dpi=200)
print("Saved: five_training_pose_single_point_dataset.png")

from matplotlib.patches import Rectangle
class MeanOnlyWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        pred = self.base(x)
        mu, _ = torch.chunk(pred, 2, dim=-1)
        return mu


class VarOnlyWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        pred = self.base(x)
        _, logvar = torch.chunk(pred, 2, dim=-1)
        logvar = torch.clamp(logvar, -20.0, 10.0)
        return torch.exp(logvar)

mean_model = MeanOnlyWrapper(model)
var_model  = VarOnlyWrapper(model)

mean_model.to(device).eval()
var_model.to(device).eval()
# _ = demo_obstacles_and_20_points(mean_model, device, robot_xy=None, n_points=20, avoid_obstacles=True)
# dense_obstacle_evaluation(mean_model, device)
# _ = demo_obstacles_and_points(
#     mean_model,
#     device,
#     n_random=60,
#     n_surface=66   # total = 100
# )

# _ = demo_visible_obstacles_and_points(
#     mean_model,
#     device,
#     robot_xy=None,
#     n_visible=100,
#     n_occluded=60,
#     n_surface_visible=40,
#     annotate=True  # set True if you want GT/Pred text on every point
# )
# _ = demo_obstacles_and_20_points(var_model, device, robot_xy=None, n_points=20, avoid_obstacles=True)
# dense_obstacle_evaluation(var_model, device)
# _ = demo_obstacles_and_points(
#     var_model,
#     device,
#     n_random=60,
#     n_surface=66   # total = 100
# )

# _ = demo_visible_obstacles_and_points(
#     var_model,
#     device,
#     robot_xy=None,
#     n_visible=100,
#     n_occluded=60,
#     n_surface_visible=40,
#     annotate=True  # set True if you want GT/Pred text on every point
# )
# _ = demo_obstacles_and_20_points(model, device, robot_xy=None, n_points=20, avoid_obstacles=True)
# dense_obstacle_evaluation(model, device)
# _ = demo_obstacles_and_points(
#     model,
#     device,
#     n_random=60,
#     n_surface=66   # total = 100
# )

# _ = demo_visible_obstacles_and_points(
#     model,
#     device,
#     robot_xy=None,
#     n_visible=100,
#     n_occluded=60,
#     n_surface_visible=40,
#     annotate=True  # set True if you want GT/Pred text on every point
# )