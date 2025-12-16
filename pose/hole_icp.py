import numpy as np
import open3d as o3d


class HoleICPRegister:
    """Register a hole STL model to an observed point cloud via ICP."""

    def __init__(self, model_path: str, sample_points: int = 5000, voxel_size: float = 0.002):
        """
        Args:
            model_path: Path to STL mesh of the hole.
            sample_points: Number of points sampled from the mesh surface.
            voxel_size: Downsample voxel size (meters).
        """
        self.model_path = model_path
        self.sample_points = sample_points
        self.voxel_size = voxel_size

        self.model_pcd = None
        self.model_ds = None
        self._load_model()

    def _load_model(self):
        mesh = o3d.io.read_triangle_mesh(self.model_path)
        if mesh.is_empty():
            raise ValueError(f"Failed to load mesh from {self.model_path}")

        mesh.compute_vertex_normals()

        # Sample points on the surface for a clean template cloud
        pcd = mesh.sample_points_poisson_disk(self.sample_points)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 4, max_nn=30))

        # Downsample for faster ICP
        pcd_ds = pcd.voxel_down_sample(self.voxel_size)
        pcd_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 3, max_nn=30))

        self.model_pcd = pcd
        self.model_ds = pcd_ds

    def align_model_z_to_normal(self, normal: np.ndarray) -> np.ndarray:
        """
        Create a rotation matrix that aligns the model's +Z axis to the given normal.
        """
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        z_axis = np.array([0.0, 0.0, 1.0])

        v = np.cross(z_axis, normal)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, normal)

        if s < 1e-6:
            # Already aligned or opposite
            if c > 0:
                return np.eye(3)
            # 180 degree rotation around X (could be any axis perpendicular to z_axis)
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        return R

    def registration(self, target_pcd: o3d.geometry.PointCloud, init_transform: np.ndarray) -> o3d.pipelines.registration.RegistrationResult:
        """
        Run point-to-plane ICP between the template and target clouds.
        """
        if target_pcd is None or len(target_pcd.points) == 0:
            raise ValueError("Target point cloud is empty.")

        target_ds = target_pcd.voxel_down_sample(self.voxel_size)
        if len(target_ds.points) == 0:
            target_ds = target_pcd

        target_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 3, max_nn=30))

        threshold = self.voxel_size * 8.0
        result = o3d.pipelines.registration.registration_icp(
            source=self.model_ds,
            target=target_ds,
            max_correspondence_distance=threshold,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
        )
        return result

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ, degrees)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        return np.degrees([roll, pitch, yaw])
