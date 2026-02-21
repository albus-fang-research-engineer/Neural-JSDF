import numpy as np
import pybullet as p
import pybullet_data
import os
import matplotlib.pyplot as plt
# ============================================================
# USER WORLD
# ============================================================

WORLD_MIN, WORLD_MAX = -2.0, 2.0


def make_3_box_obstacles():
    return [
        (-1.4, -0.6, -0.2,  0.6),
        ( 0.3,  1.2, -1.2, -0.4),
        (-0.2,  0.6,  0.5,  1.4),
    ]


# ============================================================
# PYBULLET SETUP
# ============================================================

def init_pybullet(gui=False):
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")


# ============================================================
# OBSTACLE CREATION
# ============================================================

def create_box_from_rect(rect, height=1.0):
    xmin, xmax, ymin, ymax = rect

    hx = (xmax - xmin) / 2
    hy = (ymax - ymin) / 2
    hz = height / 2

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = hz

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                              rgbaColor=[0.8, 0.3, 0.3, 1])

    p.createMultiBody(baseCollisionShapeIndex=col,
                      baseVisualShapeIndex=vis,
                      basePosition=[cx, cy, cz])


def load_obstacles():
    for rect in make_3_box_obstacles():
        create_box_from_rect(rect)


# ============================================================
# CAMERA
# ============================================================

def compute_camera_matrices(robot_pose, width, height, fov, near, far):
    x, y, yaw = robot_pose

    cam_pos = [x, y, 0.8]

    forward = np.array([np.cos(yaw), np.sin(yaw), 0])
    target = cam_pos + forward

    view = p.computeViewMatrix(
        cam_pos,
        target,
        [0, 0, 1]
    )

    proj = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near,
        farVal=far
    )

    return view, proj


# ============================================================
# DEPTH RENDER
# ============================================================

def render_depth(robot_pose,
                 width=640,
                 height=480,
                 fov=60,
                 near=0.1,
                 far=5.0):

    view, proj = compute_camera_matrices(robot_pose, width, height, fov, near, far)

    img = p.getCameraImage(width, height, view, proj,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)

    depth_buffer = np.reshape(img[3], (height, width))

    depth = far * near / (far - (far - near) * depth_buffer)

    return depth


# ============================================================
# NOISE MODEL
# ============================================================

def add_rgbd_noise(depth):

    a = 0.002
    b = 0.0015

    sigma = a + b * depth ** 2
    noisy = depth + np.random.normal(0, sigma)

    noisy = np.round(noisy / 0.001) * 0.001

    dropout = np.random.rand(*depth.shape) < 0.02
    noisy[dropout] = 0.0

    return noisy


# ============================================================
# POINT CLOUD
# ============================================================

def depth_to_pointcloud(depth, fov):

    h, w = depth.shape

    fx = fy = w / (2 * np.tan(np.deg2rad(fov) / 2))
    cx = w / 2
    cy = h / 2

    v, u = np.indices(depth.shape)

    valid = depth > 0

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    return np.stack((x, y, z), axis=1)


# ============================================================
# MAIN PIPELINE
# ============================================================

def simulate_rgbd_pointcloud(robot_pose,
                             width=640,
                             height=480,
                             fov=60):

    depth = render_depth(robot_pose, width, height, fov)

    depth_noisy = add_rgbd_noise(depth)

    pc = depth_to_pointcloud(depth_noisy, fov)

    return depth, depth_noisy, pc
def pointcloud_cam_to_world(pc_cam, robot_pose):

    x, y, yaw = robot_pose
    cam_pos = np.array([x, y, 0.8])

    # Bullet camera: Z forward, X right, Y up
    # Convert to world

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # camera optical frame → world frame
    R_cam = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ])

    R = Rz @ R_cam

    pc_world = (R @ pc_cam.T).T + cam_pos

    return pc_world
# def pointcloud_cam_to_world(pc_cam, robot_pose):
#     x, y, yaw = robot_pose

#     cam_pos = np.array([x, y, 0.8])

#     Rz = np.array([
#         [np.cos(yaw), -np.sin(yaw), 0],
#         [np.sin(yaw),  np.cos(yaw), 0],
#         [0, 0, 1]
#     ])

#     pitch = 0.0
#     Ry = np.array([
#         [ np.cos(pitch), 0, np.sin(pitch)],
#         [0, 1, 0],
#         [-np.sin(pitch), 0, np.cos(pitch)]
#     ])

#     R = Rz @ Ry

#     pc_world = (R @ pc_cam.T).T + cam_pos

#     return pc_world

def extract_planar_slice(pc_world, slice_z=0.8, thickness=0.02):
    z = pc_world[:, 2]
    mask = np.abs(z - slice_z) < thickness
    return pc_world[mask][:, :2]

def plot_2d_slice(points_2d, robot_pose, save_path):

    fig, ax = plt.subplots(figsize=(6, 6))

    # plot obstacles
    for xmin, xmax, ymin, ymax in make_3_box_obstacles():
        rect = plt.Rectangle((xmin, ymin),
                             xmax - xmin,
                             ymax - ymin,
                             fill=False,
                             linewidth=2)
        ax.add_patch(rect)

    # plot points
    if len(points_2d) > 0:
        ax.scatter(points_2d[:, 0], points_2d[:, 1],
                   s=5, c="red", label="noisy slice")

    # robot
    x, y, yaw = robot_pose
    ax.scatter(x, y, c="blue", s=80, label="robot")

    dx = 0.2 * np.cos(yaw)
    dy = 0.2 * np.sin(yaw)
    ax.arrow(x, y, dx, dy, width=0.02)

    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    plt.savefig(save_path, dpi=150)
    plt.close()
    
def draw_camera_debug(robot_pose, camera_height=0.8, ray_len=1.0):
    x, y, yaw = robot_pose

    cam_pos = np.array([x, y, camera_height])

    forward = np.array([np.cos(yaw), np.sin(yaw), 0])

    right = np.array([-np.sin(yaw), np.cos(yaw), 0])
    up = np.array([0, 0, 1])

    # forward (view direction)
    p.addUserDebugLine(cam_pos, cam_pos + ray_len * forward, [1, 0, 0], 3)

    # right axis
    p.addUserDebugLine(cam_pos, cam_pos + 0.3 * right, [0, 1, 0], 2)

    # up axis
    p.addUserDebugLine(cam_pos, cam_pos + 0.3 * up, [0, 0, 1], 2)

    # camera body (small sphere)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseVisualShapeIndex=vis, basePosition=cam_pos)
# ============================================================
#  USAGE
# ============================================================

if __name__ == "__main__":

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    init_pybullet(gui=True)
    load_obstacles()

    robot_pose = (0.0, -1.5, np.pi / 2)
    camera_height = 0.8
    draw_camera_debug(robot_pose, camera_height)
    depth_gt, depth_noisy, pc_cam = simulate_rgbd_pointcloud(robot_pose)

    # camera → world
    pc_world = pointcloud_cam_to_world(pc_cam, robot_pose)

    # extract planar slice
    pc_2d = extract_planar_slice(pc_world,
                                 slice_z=camera_height,
                                 thickness=0.1)

    # save data
    np.save(os.path.join(out_dir, "pointcloud_slice_2d.npy"), pc_2d)

    # save plot
    plot_2d_slice(
        pc_2d,
        robot_pose,
        os.path.join(out_dir, "pointcloud_slice_2d.png")
    )
    print("Saved slice with", pc_2d.shape[0], "points")
    print("Output folder:", os.path.abspath(out_dir))
    while True:
        p.stepSimulation()
