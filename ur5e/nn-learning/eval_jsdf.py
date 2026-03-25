import numpy as np
from fk_helper import RobotSDFHelper   # make sure filename matches

def main():
    # ====== CHANGE THIS PATH ======
    # urdf_path = "/home/albusfang/Albus/Neural-JSDF/ur5e/robot_models/ur5e/ur5e.urdf"
    urdf_path = "/root/Neural-JSDF/ur5e/robot_models/ur5e/ur5e.urdf"
    # Initialize robot
    robot = RobotSDFHelper(urdf_path)

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

    print("\nLink-wise distances:")
    for k, v in distances.items():
        print(f"{k}: {v:.4f}")

    # Visualize
    robot.visualize(point)


if __name__ == "__main__":
    main()