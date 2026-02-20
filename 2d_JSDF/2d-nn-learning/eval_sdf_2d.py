import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sdf.robot_sdf import RobotSdfCollisionNet


MODEL_PATH = "sdf_2d.pt"
RADIUS = 0.105


# -------------------------------------------------
# ANALYTIC GROUND TRUTH (circle SDF)
# -------------------------------------------------
def gt_signed_distance(robot_xy, points):
    diff = points - robot_xy
    return np.linalg.norm(diff, axis=1) - RADIUS


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

px = np.random.uniform(-2, 2, N)
py = np.random.uniform(-2, 2, N)

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

with torch.no_grad():
    pred = model(x_tensor).squeeze().cpu().numpy()

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


# -------------------------------------------------
# PLOT: PREDICTED vs GT
# -------------------------------------------------
plt.figure(figsize=(6, 6))

plt.scatter(gt, pred, s=2, alpha=0.3)

lims = [
    np.min([gt.min(), pred.min()]),
    np.max([gt.max(), pred.max()])
]

plt.plot(lims, lims, 'k--')  # y = x line

plt.xlabel("Ground Truth Distance [m]")
plt.ylabel("Predicted Distance [m]")
plt.title("Neural SDF Prediction vs Ground Truth")

plt.grid(True)
plt.tight_layout()
plt.savefig("pred_vs_gt.png", dpi=200)

print("Saved: pred_vs_gt.png")


NUM_POSES = 4
POINTS_PER_POSE = 4000

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax in axes.flat:

    # ----------------------------
    # random robot pose
    # ----------------------------
    robot_xy = np.random.uniform(-1.5, 1.5, size=2)

    px = np.random.uniform(-2, 2, POINTS_PER_POSE)
    py = np.random.uniform(-2, 2, POINTS_PER_POSE)
    points = np.stack([px, py], axis=1)

    gt = gt_signed_distance(robot_xy, points)

    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], POINTS_PER_POSE, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        pred = model(
            torch.tensor(x_input, dtype=torch.float32, device=device)
        ).cpu().numpy().squeeze()

    error = np.abs(pred - gt)

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
    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.set_title(f"Robot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

fig.colorbar(sc, ax=axes.ravel().tolist(), location="right", label="|Prediction Error| [m]")
# plt.tight_layout()
plt.savefig("spatial_error.png", dpi=200)

# -------------------------------------------------
# QUALITATIVE CHECK: 5 poses, 1 point each
# -------------------------------------------------
NUM_TEST = 5

fig, axes = plt.subplots(1, NUM_TEST, figsize=(4 * NUM_TEST, 4))

if NUM_TEST == 1:
    axes = [axes]

for ax in axes:

    # random robot pose
    robot_xy = np.random.uniform(-1.5, 1.5, size=2)

    # one random query point
    point = np.random.uniform(-2, 2, size=(1, 2))

    # ground truth
    gt = gt_signed_distance(robot_xy, point)[0]

    # model input
    x_input = np.concatenate(
        [robot_xy[None, :], point],
        axis=1
    )

    with torch.no_grad():
        pred = model(
            torch.tensor(x_input, dtype=torch.float32, device=device)
        ).cpu().numpy().squeeze()

    # ----------------------------
    # PLOT
    # ----------------------------
    ax.scatter(point[0, 0], point[0, 1], s=80)

    circle = plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2)
    ax.add_patch(circle)

    ax.text(
        point[0, 0],
        point[0, 1],
        f"GT: {gt:.3f}\nPred: {pred:.3f}",
        fontsize=9,
        verticalalignment="bottom"
    )

    ax.set_title(f"Robot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

plt.tight_layout()
plt.savefig("five_random_pose_single_point.png", dpi=200)
print("Saved: five_random_pose_single_point.png")