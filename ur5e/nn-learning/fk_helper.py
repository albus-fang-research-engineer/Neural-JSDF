import numpy as np
from urdfpy import URDF
import trimesh
import open3d as o3d


class RobotSDFHelper:
    def __init__(self, urdf_path):
        self.robot = URDF.load(urdf_path)

        # store joints
        self.joint_names = [
            j.name for j in self.robot.joints
            if j.joint_type in ["revolute", "prismatic", "continuous"]
        ]

        self.num_joints = len(self.joint_names)

        # initialize config
        self._robot_joints = {name: 0.0 for name in self.joint_names}

        # build link meshes (LOCAL frame)
        self.link_meshes = {}
        self.link_names = []

        for link in self.robot.links:
            if not link.visuals:
                continue

            meshes = []
            for visual in link.visuals:
                for m in visual.geometry.meshes:
                    mesh = m.copy()

                    if visual.origin is not None:
                        mesh.apply_transform(visual.origin)

                    meshes.append(mesh)

            if meshes:
                combined = trimesh.util.concatenate(meshes)
                self.link_meshes[link.name] = combined
                self.link_names.append(link.name)

    # ------------------------
    # Set configuration
    # ------------------------
    def set_q(self, q):
        assert len(q) == self.num_joints
        for i, name in enumerate(self.joint_names):
            self._robot_joints[name] = q[i]

    # ------------------------
    # FK → world meshes
    # ------------------------
    def get_link_meshes_world(self):
        fk = self.robot.link_fk(cfg=self._robot_joints)

        meshes_world = {}

        for name in self.link_names:
            mesh = self.link_meshes[name].copy()
            link_obj = self.robot.link_map[name]
            mesh.apply_transform(fk[link_obj])
            meshes_world[name] = mesh

        return meshes_world

    # ------------------------
    # Distance query
    # ------------------------
    def compute_link_distances(self, point):
        """
        point: (3,)
        returns: dict {link_name: distance}
        """
        meshes = self.get_link_meshes_world()

        distances = {}
        for name, mesh in meshes.items():
            # closest point on mesh
            closest, dist, _ = trimesh.proximity.closest_point(mesh, [point])
            distances[name] = dist[0]

        return distances

    # ------------------------
    # Visualization
    # ------------------------
    def visualize(self, point=None):
        meshes = self.get_link_meshes_world()

        geometries = []

        for mesh in meshes.values():
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )
            o3d_mesh.compute_vertex_normals()
            geometries.append(o3d_mesh)

        if point is not None:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])
            geometries.append(sphere)

        o3d.visualization.draw_geometries(geometries)
    
    def get_ee_position(self):
        fk = self.robot.link_fk(cfg=self._robot_joints)

        # Use correct EE link
        ee_link = self.robot.link_map["wrist_3_link"]  # or "tool0" if available

        ee_transform = fk[ee_link]
        return ee_transform[:3, 3]