import numpy as np
from fk_helper import RobotSDFHelper   # make sure filename matches
from njsdf_loader import NJSDFHelper
def main():
    # ====== CHANGE THIS PATH ======
    # urdf_path = "/home/albusfang/Albus/Neural-JSDF/ur5e/robot_models/ur5e/ur5e.urdf"
    urdf_path = "/root/Neural-JSDF/ur5e/robot_models/ur5e/ur5e.urdf"
    # Initialize robot
    robot = RobotSDFHelper(urdf_path)


    # ====== LOAD N-JSDF MODEL ======
    model_path = "/root/Neural-JSDF/ur5e/nn-learning/ur5e_sdf_256x5_mesh.pt"  # change if needed

    njsdf = NJSDFHelper(
        model_path=model_path,
        num_joints=robot.num_joints,
        device="cuda"  # or "cpu"
    )

    # Sample random joint config
    q = np.random.uniform(-1.0, 1.0, robot.num_joints)
    robot.set_q(q)

    print("Random configuration q:")
    print(q)

    # Sample random 3D point (adjust range if needed)
    point = np.random.uniform(low=[-0.5, -0.5, 0.0],
                              high=[0.5, 0.5, 0.8])

    print("Random query point:")
    print(point)

    # Compute distances
    distances = robot.compute_link_distances(point)
    # ====== Neural prediction ======
    pred_dist = njsdf.predict(q, point).squeeze()  # shape (num_links,)

    print("\nLink-wise distances:")
    for k, v in distances.items():
        print(f"{k}: {v:.4f}")
    print("\n=== Link-wise distance comparison ===")

    gt_vals = []
    pred_vals = []

    for i, (k, v) in enumerate(distances.items()):
        gt = v
        pred = pred_dist[i]

        gt_vals.append(gt)
        pred_vals.append(pred)

        print(f"{k}: GT = {gt:.4f} m | Pred = {pred:.4f} m | Error = {abs(gt - pred):.4f}")

    # Optional summary
    gt_vals = np.array(gt_vals)
    pred_vals = np.array(pred_vals)

    print("\n=== Summary ===")
    print(f"Mean abs error: {np.mean(np.abs(gt_vals - pred_vals)):.4f} m")
    print(f"Max abs error:  {np.max(np.abs(gt_vals - pred_vals)):.4f} m")
    print(f"Min GT dist:    {np.min(gt_vals):.4f} m")
    print(f"Min Pred dist:  {np.min(pred_vals):.4f} m")
    # ====== MULTI-POINT EVALUATION ======
    num_points = 200

    points = np.random.uniform(
        low=[-0.5, -0.5, 0.2],
        high=[0.5, 0.5, 0.8],
        size=(num_points, 3)
    )

    all_gt = []
    all_pred = []

    print(f"\nEvaluating {num_points} random points...\n")

    for idx, point in enumerate(points):

        distances = robot.compute_link_distances(point)
        pred_dist = njsdf.predict(q, point).squeeze()

        for i, (k, v) in enumerate(distances.items()):
            gt = v
            pred = pred_dist[i]

            all_gt.append(gt)
            all_pred.append(pred)

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx+1}/{num_points} points")

    # Convert to arrays
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)

    errors = np.abs(all_gt - all_pred)

    print("\n=== Aggregate Results ===")
    print(f"Total samples: {len(errors)}  (points × links)")
    print(f"Mean abs error: {np.mean(errors):.6f} m")
    print(f"Median abs error: {np.median(errors):.6f} m")
    print(f"Max abs error: {np.max(errors):.6f} m")

    print("\n=== Distance Stats ===")
    print(f"Min GT dist: {np.min(all_gt):.6f} m")
    print(f"Min Pred dist: {np.min(all_pred):.6f} m")
    
    # Visualize
    # robot.visualize(point)-


if __name__ == "__main__":
    main()