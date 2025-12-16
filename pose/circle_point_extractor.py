import numpy as np
import open3d as o3d


class CirclePointCloudExtractor:
    def __init__(self, margin=30, depth_scale=None):
        """
        Args:
            margin: 圆周围的额外像素边距
            depth_scale: 深度缩放因子（如果为None则自动获取）
        """
        self.margin = margin
        self.depth_scale = depth_scale
        self.circle_center_3d = None

    def get_circle_center_3d(self, circle ,depth_image, camera):
        """
         计算圆心的3D坐标

         Args:
             circle: (cx, cy, r)
             depth_image: 深度图
             camera: RealSenseCamera实例

         Returns:
             np.array([x, y, z]) 或 None
         """
        if circle is None:
            return None
        cx, cy, r = map(int, circle)

        # 在圆心周围取深度中值（更鲁棒）
        radius_sample = max(3, int(r * 0.3))
        y_min = max(0, cy - radius_sample)
        y_max = min(depth_image.shape[0], cy + radius_sample)
        x_min = max(0, cx - radius_sample)
        x_max = min(depth_image.shape[1], cx + radius_sample)

        depth_region = depth_image[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0]

        if len(valid_depths) == 0:
            return None

        # 使用中值深度
        depth_value = np.median(valid_depths)

        # 获取depth_scale
        if self.depth_scale is None:
            self.depth_scale = camera.depth_scale

        z = depth_value * self.depth_scale

        if z <= 0 or z > 2.5:
            return None

        # 获取相机内参
        intrinsics = camera.get_intrinsics()
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx_intr = intrinsics.intrinsic_matrix[0, 2]
        cy_intr = intrinsics.intrinsic_matrix[1, 2]

        # 反投影到3D
        x = (cx - cx_intr) * z / fx
        y = (cy - cy_intr) * z / fy

        # 应用坐标变换（与点云一致）
        center_3d = np.array([x, y, z])

        # 保存到实例变量
        self.circle_center_3d = center_3d

        return center_3d

    def extract(self, pcd_full, circle, color_image, depth_image, intrinsics, camera=None):
        """
        从完整点云中提取圆形区域的点云

        Args:
            pcd_full: 完整的点云（Open3D PointCloud）
            circle: 检测到的圆 (cx, cy, r)
            color_image: 彩色图像
            depth_image: 深度图像
            intrinsics: 相机内参 Open3D PinholeCameraIntrinsic
            camera: RealSenseCamera实例（用于获取depth_scale）

        Returns:
            pcd_circle: 提取的圆形区域点云
        """
        if circle is None:
            return None

        # 获取depth_scale
        if self.depth_scale is None and camera is not None:
            self.depth_scale = camera.depth_scale

        if self.depth_scale is None:
            self.depth_scale = 0.001  # 默认值

        cx, cy, r = map(int, circle)

        # 计算ROI范围
        height, width = depth_image.shape
        x_min = max(0, cx - r - self.margin)
        x_max = min(width, cx + r + self.margin)
        y_min = max(0, cy - r - self.margin)
        y_max = min(height, cy + r + self.margin)

        # 创建圆形mask（向量化操作，更快）
        y_coords, x_coords = np.ogrid[0:height, 0:width]
        dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        mask = dist_from_center <= (r + self.margin)

        # 提取点云（手动方式，更可控）
        points = []
        colors = []

        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx_intr = intrinsics.intrinsic_matrix[0, 2]
        cy_intr = intrinsics.intrinsic_matrix[1, 2]

        for v in range(y_min, y_max):
            for u in range(x_min, x_max):
                if not mask[v, u]:
                    continue

                depth_value = depth_image[v, u]
                if depth_value == 0:
                    continue

                # 转换深度值（毫米转米）
                z = depth_value * self.depth_scale

                if z <= 0 or z > 2.5:  # 深度范围检查
                    continue

                # 反投影到3D空间
                x = (u - cx_intr) * z / fx
                y = (v - cy_intr) * z / fy

                points.append([x, -y, -z])  # 直接应用坐标变换

                # 获取颜色（BGR转RGB，归一化）
                b, g, r_color = color_image[v, u]
                colors.append([r_color / 255.0, g / 255.0, b / 255.0])

        # 创建点云
        pcd_circle = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd_circle.points = o3d.utility.Vector3dVector(np.array(points))
            pcd_circle.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd_circle

