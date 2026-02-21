from matplotlib.patches import Rectangle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
RADIUS = 0.105
def gt_signed_distance(robot_xy, points):
    diff = points - robot_xy
    return np.linalg.norm(diff, axis=1) - RADIUS
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
    plt.savefig("deterministic_plots/obstacles_100pts_gt_pred.png", dpi=200)
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
    plt.savefig("deterministic_plots/obstacles_20pts_gt_pred.png", dpi=200)
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
    plt.savefig("deterministic_plots/dense_obstacle_error.png", dpi=200)

    print("Dense MAE:", np.mean(error))
    print("Dense RMSE:", np.sqrt(np.mean((pred - gt) ** 2)))

def segment_intersects_box(p0, p1, box):
    """
    p0 → robot
    p1 → query point
    box = (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = box

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    tmin = 0.0
    tmax = 1.0

    for p, d, bmin, bmax in [
        (p0[0], dx, xmin, xmax),
        (p0[1], dy, ymin, ymax),
    ]:
        if abs(d) < 1e-8:
            if p < bmin or p > bmax:
                return False
        else:
            t1 = (bmin - p) / d
            t2 = (bmax - p) / d
            t_enter = min(t1, t2)
            t_exit  = max(t1, t2)

            tmin = max(tmin, t_enter)
            tmax = min(tmax, t_exit)

            if tmin > tmax:
                return False

    return True
def sample_occluded_points(robot_xy, obstacles, n_points, low=-2, high=2, max_tries=200000):
    """
    For visualization: points that are NOT visible (blocked by at least one obstacle),
    and not inside any obstacle.
    """
    pts = []
    tries = 0
    while len(pts) < n_points and tries < max_tries:
        p = np.random.uniform(low, high, size=2)
        if point_in_any_box(p, obstacles):
            tries += 1
            continue
        if not is_visible(robot_xy, p, obstacles):
            pts.append(p)
        tries += 1
    if len(pts) < n_points:
        raise RuntimeError(f"Could only sample {len(pts)}/{n_points} occluded points after {tries} tries.")
    return np.array(pts)

def is_visible(robot_xy, point, obstacles):

    for box in obstacles:
        if segment_intersects_box(robot_xy, point, box):
            return False

    return True

def sample_visible_points(robot_xy,
                          obstacles,
                          n_points,
                          low=-2,
                          high=2,
                          max_tries=200000):

    pts = []
    tries = 0

    while len(pts) < n_points and tries < max_tries:

        p = np.random.uniform(low, high, size=2)

        if point_in_any_box(p, obstacles):
            tries += 1
            continue

        if is_visible(robot_xy, p, obstacles):
            pts.append(p)

        tries += 1

    if len(pts) < n_points:
        raise RuntimeError("Could not sample enough visible points")

    return np.array(pts)

def plot_visibility(robot_xy, obstacles, visible_pts, blocked_pts):

    fig, ax = plt.subplots(figsize=(8, 8))

    for box in obstacles:
        xmin, xmax, ymin, ymax = box
        ax.add_patch(Rectangle((xmin, ymin),
                               xmax - xmin,
                               ymax - ymin,
                               fill=False,
                               linewidth=2))

    ax.add_patch(plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2))

    if len(visible_pts):
        ax.scatter(visible_pts[:, 0], visible_pts[:, 1],
                   c="green", s=5, label="visible")

    if len(blocked_pts):
        ax.scatter(blocked_pts[:, 0], blocked_pts[:, 1],
                   c="red", s=5, label="occluded")

    ax.legend()
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.grid(True)

def demo_visible_obstacles_and_points(model, device,
                                      robot_xy=None,
                                      n_visible=100,
                                      n_occluded=60,
                                      n_surface_visible=40,
                                      low=-2,
                                      high=2,
                                      annotate=True,
                                      save_path="deterministic_plots/visible_obstacles_points_gt_pred.png",
                                      max_tries=400000):
    """
    Plots obstacles + robot + points, but ONLY evaluates/plots GT+Pred for VISIBLE points.
    Also plots occluded points in a different marker/color so you can see the visibility.
    Visible points are colored by |pred - gt|.
    """

    obstacles = make_3_box_obstacles()

    # choose robot pose not inside obstacles
    if robot_xy is None:
        for _ in range(5000):
            candidate = np.random.uniform(-1.5, 1.5, size=2)
            if not point_in_any_box(candidate, obstacles):
                robot_xy = candidate
                break
        if robot_xy is None:
            robot_xy = np.array([0.0, 0.0])

    # visible points: mix of random visible + visible on obstacle surfaces (if possible)
    n_vis_random = max(0, n_visible - n_surface_visible)

    vis_random = sample_visible_points(
        robot_xy, obstacles, n_vis_random, low=low, high=high, max_tries=max_tries
    )

    if n_surface_visible > 0:
        # oversample edges then filter by visibility
        surf_cand = sample_box_surface(obstacles, samples_per_edge=300)  # plenty of candidates
        # remove those that are "inside" (edges are on boundary; point_in_any_box treats boundary as inside,
        # so we skip that check here and only test visibility)
        visible_mask = np.array([is_visible(robot_xy, p, obstacles) for p in surf_cand], dtype=bool)
        surf_vis = surf_cand[visible_mask]
        if len(surf_vis) == 0:
            vis_surface = np.zeros((0, 2))
        else:
            if len(surf_vis) >= n_surface_visible:
                idx = np.random.choice(len(surf_vis), size=n_surface_visible, replace=False)
            else:
                idx = np.random.choice(len(surf_vis), size=n_surface_visible, replace=True)
            vis_surface = surf_vis[idx]
    else:
        vis_surface = np.zeros((0, 2))

    visible_points = np.vstack([vis_random, vis_surface])
    N = len(visible_points)

    # optionally sample occluded points just to visualize the blocking
    if n_occluded > 0:
        occluded_points = sample_occluded_points(
            robot_xy, obstacles, n_occluded, low=low, high=high, max_tries=max_tries
        )
    else:
        occluded_points = np.zeros((0, 2))

    # GT + Pred only on visible points (as requested)
    gt = gt_signed_distance(robot_xy, visible_points)

    x_input = np.concatenate(
        [np.repeat(robot_xy[None, :], N, axis=0), visible_points],
        axis=1
    )

    with torch.no_grad():
        pred = model(torch.tensor(x_input, dtype=torch.float32, device=device)).squeeze().cpu().numpy()

    error = np.abs(pred - gt)

    # ----------------------------
    # plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # obstacles
    for (xmin, xmax, ymin, ymax) in obstacles:
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=2))

    # robot
    ax.add_patch(plt.Circle(robot_xy, RADIUS, fill=False, linewidth=2))

    # occluded points (visual only)
    if len(occluded_points) > 0:
        ax.scatter(
            occluded_points[:, 0],
            occluded_points[:, 1],
            s=20,
            marker="x",
            label="occluded (not evaluated)"
        )
        # optionally draw a few rays so it's obvious what "occluded" means
        draw_rays = min(20, len(occluded_points))
        for p in occluded_points[:draw_rays]:
            ax.plot([robot_xy[0], p[0]], [robot_xy[1], p[1]], linewidth=0.5, alpha=0.25)

    # visible points colored by error
    sc = ax.scatter(
        visible_points[:, 0],
        visible_points[:, 1],
        c=error,
        s=30,
        cmap="coolwarm",
        label="visible (evaluated)"
    )

    if annotate:
        for i in range(N):
            ax.text(
                visible_points[i, 0],
                visible_points[i, 1],
                f"{gt[i]:.2f}/{pred[i]:.2f}",
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="left"
            )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("|pred - gt| [m]")

    ax.set_title(
        f"Visible-only evaluation: N={N} (random+surface), occluded shown={len(occluded_points)}\n"
        f"Robot @ {robot_xy.round(2)}"
    )
    ax.set_aspect("equal")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")

    # quick stats
    print("Visible MAE:", float(np.mean(error)))
    print("Visible RMSE:", float(np.sqrt(np.mean((pred - gt) ** 2))))

    return robot_xy, visible_points, gt, pred, occluded_points