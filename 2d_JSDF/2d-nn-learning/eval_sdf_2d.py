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


from matplotlib.patches import Rectangle

# -------------------------------------------------
# OBSTACLES (2D axis-aligned boxes)
# -------------------------------------------------
def make_3_box_obstacles():
    """
    Returns 3 axis-aligned rectangles in the form:
      [(xmin, xmax, ymin, ymax), ...]
    Coordinates are in the same world frame as robot_xy and points.
    """
    obstacles = [
        (-1.4, -0.6, -0.2,  0.6),  # left box
        ( 0.3,  1.2, -1.2, -0.4),  # bottom-right box
        (-0.2,  0.6,  0.5,  1.4),  # top-middle box
    ]
    return obstacles


def point_in_any_box(pt_xy, obstacles):
    x, y = float(pt_xy[0]), float(pt_xy[1])
    for (xmin, xmax, ymin, ymax) in obstacles:
        if (xmin <= x <= xmax) and (ymin <= y <= ymax):
            return True
    return False


def sample_points_avoiding_boxes(n, obstacles, low=-2.0, high=2.0, max_tries=200000):
    """
    Uniformly sample points in [low, high]^2 while rejecting those inside any obstacle.
    """
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = np.random.uniform(low, high, size=(2,))
        if not point_in_any_box(p, obstacles):
            pts.append(p)
        tries += 1

    if len(pts) < n:
        raise RuntimeError(f"Could only sample {len(pts)}/{n} points outside obstacles after {tries} tries.")
    return np.array(pts)

def sample_box_surface(obstacles, samples_per_edge=200):
    pts = []

    for (xmin, xmax, ymin, ymax) in obstacles:

        xs = np.random.uniform(xmin, xmax, samples_per_edge)
        ys = np.random.uniform(ymin, ymax, samples_per_edge)

        # bottom edge
        pts.append(np.stack([xs, np.full_like(xs, ymin)], axis=1))

        # top edge
        pts.append(np.stack([xs, np.full_like(xs, ymax)], axis=1))

        # left edge
        pts.append(np.stack([np.full_like(ys, xmin), ys], axis=1))

        # right edge
        pts.append(np.stack([np.full_like(ys, xmax), ys], axis=1))

    return np.concatenate(pts, axis=0)

def demo_obstacles_and_points(model, device,
                              robot_xy=None,
                              n_random=60,
                              n_surface=40):

    obstacles = make_3_box_obstacles()

    # pick robot pose away from boxes
    if robot_xy is None:
        while True:
            candidate = np.random.uniform(-1.5, 1.5, size=2)
            if not point_in_any_box(candidate, obstacles):
                robot_xy = candidate
                break

    # ----------------------------
    # sample random free-space points
    # ----------------------------
    random_pts = sample_points_avoiding_boxes(
        n_random, obstacles, low=-2, high=2
    )

    # ----------------------------
    # sample obstacle surface points
    # ----------------------------
    surface_pts = sample_box_surface(
        obstacles,
        samples_per_edge=max(1, n_surface // (len(obstacles) * 4))
    )

    surface_pts = surface_pts[:n_surface]

    points = np.vstack([random_pts, surface_pts])
    N = len(points)

    # ----------------------------
    # GT
    # ----------------------------
    gt = gt_signed_distance(robot_xy, points)

    # ----------------------------
    # NN inference
    # ----------------------------
    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], N, axis=0), points],
        axis=1
    )

    with torch.no_grad():
        pred = model(
            torch.tensor(x_input, dtype=torch.float32, device=device)
        ).cpu().numpy().squeeze()

    error = np.abs(pred - gt)

    # ----------------------------
    # PRINT TABLE
    # ----------------------------
    print(f"\n--- {N}-point check (GT vs Pred) ---")
    print(" idx |   px     py   |    GT     Pred    |  |err|")
    print("-----+--------------+--------------------+--------")

    for i in range(N):
        print(f"{i:4d} | {points[i,0]:6.3f} {points[i,1]:6.3f} | "
              f"{gt[i]:7.4f} {pred[i]:7.4f} | {error[i]:6.4f}")

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        ax.add_patch(Rectangle((xmin, ymin),
                               xmax - xmin,
                               ymax - ymin,
                               fill=False,
                               linewidth=2))

    # robot
    ax.add_patch(plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2))

    # color by error
    sc = ax.scatter(points[:, 0],
                    points[:, 1],
                    c=error,
                    cmap="coolwarm",
                    s=35)

    for i in range(N):
        ax.text(points[i, 0],
                points[i, 1],
                f"{gt[i]:.2f}/{pred[i]:.2f}",
                fontsize=7)

    plt.colorbar(sc, label="|error|")

    ax.set_title(f"{N} points (random + obstacle surface)\nRobot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("obstacles_100pts_gt_pred.png", dpi=200)
    print("Saved: obstacles_100pts_gt_pred.png")

    return robot_xy, points, gt, pred
# -------------------------------------------------
# DEMO: obstacles + 20 points, show GT vs Pred
# -------------------------------------------------
def demo_obstacles_and_20_points(model, device, robot_xy=None, n_points=20, avoid_obstacles=True):
    obstacles = make_3_box_obstacles()

    # choose a robot pose (keep it away from boxes for a clean visualization)
    if robot_xy is None:
        for _ in range(1000):
            candidate = np.random.uniform(-1.5, 1.5, size=2)
            if not point_in_any_box(candidate, obstacles):
                robot_xy = candidate
                break
        if robot_xy is None:
            robot_xy = np.array([0.0, 0.0])

    # sample points
    if avoid_obstacles:
        points = sample_points_avoiding_boxes(n_points, obstacles, low=-2.0, high=2.0)
    else:
        points = np.random.uniform(-2, 2, size=(n_points, 2))

    # GT
    gt = gt_signed_distance(robot_xy, points)

    # Pred
    x_input = np.concatenate([np.repeat(robot_xy[None, :], n_points, axis=0), points], axis=1)
    with torch.no_grad():
        pred = model(torch.tensor(x_input, dtype=torch.float32, device=device)).squeeze().cpu().numpy()

    # Print table
    print("\n--- 20-point check (GT vs Pred) ---")
    print(" idx |   px     py   |    GT     Pred    |  |err|")
    print("-----+--------------+--------------------+--------")
    for i in range(n_points):
        err = abs(float(pred[i]) - float(gt[i]))
        print(f"{i:4d} | {points[i,0]:6.3f} {points[i,1]:6.3f} | {gt[i]:7.4f} {pred[i]:7.4f} | {err:6.4f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # draw obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=2))

    # draw robot footprint
    ax.add_patch(plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2))

    # draw points
    ax.scatter(points[:, 0], points[:, 1], s=35)

    # annotate each point with GT + Pred
    for i in range(n_points):
        ax.text(
            points[i, 0],
            points[i, 1],
            f"{gt[i]:.2f}/{pred[i]:.2f}",  # GT/Pred
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="left"
        )

    ax.set_title(f"Obstacles + {n_points} points (annotated GT/Pred)\nRobot @ {robot_xy.round(2)}")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("obstacles_20pts_gt_pred.png", dpi=200)
    print("Saved: obstacles_20pts_gt_pred.png")

    return robot_xy, points, gt, pred

def dense_obstacle_evaluation(model, device,
                              N_random=4000,
                              N_surface_per_edge=300):

    obstacles = make_3_box_obstacles()

    # safe robot pose
    while True:
        robot_xy = np.random.uniform(-1.5, 1.5, size=2)
        if not point_in_any_box(robot_xy, obstacles):
            break

    # random free-space points
    random_pts = sample_points_avoiding_boxes(
        N_random, obstacles, low=-2, high=2
    )

    # obstacle surface points
    surface_pts = sample_box_surface(
        obstacles,
        samples_per_edge=N_surface_per_edge
    )

    points = np.vstack([random_pts, surface_pts])

    # GT
    gt = gt_signed_distance(robot_xy, points)

    # NN inference
    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], len(points), axis=0), points],
        axis=1
    )

    with torch.no_grad():
        pred = model(
            torch.tensor(x_input, dtype=torch.float32, device=device)
        ).cpu().numpy().squeeze()

    error = np.abs(pred - gt)

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        ax.add_patch(Rectangle((xmin, ymin),
                               xmax - xmin,
                               ymax - ymin,
                               fill=False,
                               linewidth=2))

    # robot
    ax.add_patch(plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2))

    sc = ax.scatter(points[:, 0],
                    points[:, 1],
                    c=error,
                    s=3,
                    cmap="coolwarm")

    plt.colorbar(sc, label="|SDF error|")

    ax.set_title("Dense obstacle evaluation (surface oversampled)")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("dense_obstacle_error.png", dpi=200)

    print("Dense MAE:", np.mean(error))
    print("Dense RMSE:", np.sqrt(np.mean((pred - gt) ** 2)))

_ = demo_obstacles_and_20_points(model, device, robot_xy=None, n_points=20, avoid_obstacles=True)
dense_obstacle_evaluation(model, device)
_ = demo_obstacles_and_points(
    model,
    device,
    n_random=60,
    n_surface=66   # total = 100
)