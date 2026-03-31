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

    model_path = "/root/Neural-JSDF/ur5e/nn-learning/ur5e_sdf_256x5_mesh_4000_configs.pt"  # change if needed

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
    
    # ====== MULTI-CONFIG + MULTI-POINT EVALUATION ======
    num_configs = 50
    num_points_per_config = 100

    all_gt = []
    all_pred = []

    print(f"\nEvaluating {num_configs} configs × {num_points_per_config} points...\n")
    records = []
    for c in range(num_configs):

        # Sample random configuration
        q = np.random.uniform(-1.0, 1.0, robot.num_joints)
        robot.set_q(q)
        
        ee_pos = robot.get_ee_position()
        if ee_pos[2] < 0.2:
            continue
        # Sample points for this config
        points = np.random.uniform(
            low=[-0.5, -0.5, 0.0],
            high=[0.5, 0.5, 0.8],
            size=(num_points_per_config, 3)
        )

        for point in points:

            distances = robot.compute_link_distances(point)
            pred_dist = njsdf.predict(q, point).squeeze()

            for i, (k, v) in enumerate(distances.items()):
                gt = v
                pred = pred_dist[i]
                err = abs(gt-pred)
                all_gt.append(gt)
                all_pred.append(pred)
                records.append({
                    "error": err,
                    "gt": gt,
                    "pred": pred,
                    "config": q.copy(),
                    "point": point.copy(),
                    "link": k
                })

        if (c + 1) % 5 == 0:
            print(f"Processed {c+1}/{num_configs} configs")

    # Convert to arrays
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)

    errors = np.abs(all_gt - all_pred)

    print("\n=== Aggregate Results ===")
    print(f"Total samples: {len(errors)}  (configs × points × links)")
    print(f"Mean abs error: {np.mean(errors):.6f} m")
    print(f"Median abs error: {np.median(errors):.6f} m")
    print(f"Max abs error: {np.max(errors):.6f} m")

    print("\n=== Distance Stats ===")
    print(f"Min GT dist: {np.min(all_gt):.6f} m")
    print(f"Min Pred dist: {np.min(all_pred):.6f} m")
    
    # Visualize
    # robot.visualize(point)-
    # ====== QUANTILES ======
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]

    print("\n=== Error Quantiles ===")
    for q in quantiles:
        val = np.quantile(errors, q)
        print(f"{int(q*100)}th percentile: {val:.6f} m")
    # Top-K worst errors (tail behavior)
    k = int(0.01 * len(errors))  # top 1%
    top_k_errors = np.sort(errors)[-k:]

    print("\n=== Tail Error Stats (Top 1%) ===")
    print(f"Mean (top 1%): {np.mean(top_k_errors):.6f} m")
    print(f"Min (top 1%):  {np.min(top_k_errors):.6f} m")
    print(f"Max (top 1%):  {np.max(top_k_errors):.6f} m")
    # Sort by error descending
    records_sorted = sorted(records, key=lambda x: x["error"], reverse=True)

    K = 10
    top_k = records_sorted[:K]
    print("\n=== Top-K Worst Errors ===")
    for i, r in enumerate(top_k):
        print(f"\{i+1}")
        print(f"Error: {r['error']:.6f}")
        print(f"GT: {r['gt']:.6f}, Pred: {r['pred']:.6f}")
        print(f"Link: {r['link']}")
        print(f"Point: {r['point']}")
        print(f"Config (q): {r['config']}")
    high_err = records_sorted[0]

    robot.set_q(high_err["config"])
    robot.visualize(high_err["point"])
    import matplotlib.pyplot as plt

    plt.scatter(all_gt, errors, s=1)
    plt.xlabel("GT Distance")
    plt.ylabel("Error")
    plt.show()

if __name__ == "__main__":
    main()