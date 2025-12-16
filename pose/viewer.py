"""
统一的AprilTag + 圆孔检测系统
整合了6D位姿估计和3D点云可视化
"""

import cv2
import open3d as o3d
import numpy as np
import threading
import queue
import time
from realsense.rs_camera import RealSenseCamera
from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor


class AprilTagPoseEstimator:
    """AprilTag 6D位姿估计"""

    def __init__(self, camera, tag_size: float, rod_offset: np.ndarray, rod_rotation: np.ndarray = None):
        self.camera = camera
        self.tag_size = tag_size
        self.rod_offset = rod_offset.reshape(3, 1)
        self.rod_rotation = np.eye(3) if rod_rotation is None else rod_rotation

        self.K = camera.get_camera_matrix()
        self.dist = camera.get_distortion_coeffs()

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        s = tag_size / 2.0
        self.obj_points = np.array([
            [-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]
        ], dtype=np.float32)

    def detect_tag(self, image):
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            return None, None
        return corners[0].reshape(4, 2).astype(np.float32), int(ids[0][0])

    def estimate_pose(self, corners):
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points, corners, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if ok else (None, None)

    def compute_rod_tip_6d(self, rvec, tvec):
        R_tag, _ = cv2.Rodrigues(rvec)
        rod_position = tvec + R_tag @ self.rod_offset
        R_rod = R_tag @ self.rod_rotation
        rod_rvec, _ = cv2.Rodrigues(R_rod)
        return rod_position, rod_rvec

    def rotation_matrix_to_euler(self, R):
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

    def draw_on_image(self, image, corners, rvec, tvec, rod_position, rod_rvec):
        """在图像上绘制AprilTag检测结果"""
        # 绘制标签框
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        # 绘制角点
        for i, c in enumerate(corners):
            x, y = int(c[0]), int(c[1])
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            cv2.putText(image, str(i), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制标签坐标轴
        cv2.drawFrameAxes(image, self.K, self.dist, rvec, tvec, self.tag_size, 2)

        # 绘制棒子末端
        rod_tip_2d, _ = cv2.projectPoints(
            rod_position.reshape(1, 3),
            np.zeros((3, 1)), np.zeros((3, 1)),
            self.K, self.dist
        )
        rod_tip_2d = tuple(rod_tip_2d[0][0].astype(int))
        cv2.circle(image, rod_tip_2d, 6, (0, 255, 255), -1)
        cv2.putText(image, "Rod Tip", (rod_tip_2d[0] + 10, rod_tip_2d[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 绘制棒子坐标轴
        axis_length = self.tag_size * 1.5
        cv2.drawFrameAxes(image, self.K, self.dist, rod_rvec, rod_position, axis_length, 2)

        return image


class IntegratedViewer:
    """整合的可视化系统"""

    def __init__(self, circle_extractor, circle_detector, apriltag_estimator):
        self.circle_extractor = circle_extractor
        self.circle_detector = circle_detector
        self.apriltag_estimator = apriltag_estimator

        # 点云队列
        self.pcd_queue = queue.Queue(maxsize=2)
        self.running = False
        self.vis_thread = None

        # 3D几何体
        self.circle_center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        self.circle_center_sphere.paint_uniform_color([1, 0, 0])  # 红色

        self.circle_center_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.02, origin=[0, 0, 0]
        )

        # AprilTag棒子末端
        self.rod_tip_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        self.rod_tip_sphere.paint_uniform_color([0, 1, 1])  # 青色

        self.rod_tip_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.025, origin=[0, 0, 0]
        )

    def _visualizer_thread(self):
        """Open3D可视化线程"""
        vis = o3d.visualization.Visualizer()
        vis.create_window("Integrated 3D Visualization", width=1000, height=700)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        # 添加世界坐标系
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate)

        # 初始化点云
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 添加圆心几何体
        vis.add_geometry(self.circle_center_sphere)
        vis.add_geometry(self.circle_center_axes)

        # 添加棒子末端几何体
        vis.add_geometry(self.rod_tip_sphere)
        vis.add_geometry(self.rod_tip_axes)

        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])

        to_reset = True
        print("3D Visualization started.")

        while self.running:
            try:
                update_data = self.pcd_queue.get(timeout=0.1)
                if update_data is None:
                    break

                new_pcd, circle_center, rod_position = update_data

                # 更新点云
                if new_pcd is not None:
                    pcd.points = new_pcd.points
                    pcd.colors = new_pcd.colors
                    vis.update_geometry(pcd)

                # 更新圆心位置
                if circle_center is not None:
                    center = np.array([circle_center[0], -circle_center[1], -circle_center[2]])
                    self.circle_center_sphere.translate(center, relative=False)
                    self.circle_center_axes.translate(center, relative=False)
                    vis.update_geometry(self.circle_center_sphere)
                    vis.update_geometry(self.circle_center_axes)

                # 更新棒子末端位置
                if rod_position is not None:
                    # 转换坐标系（与点云一致）
                    rod_pos = np.array([
                        rod_position[0],
                        -rod_position[1],
                        -rod_position[2]
                    ])
                    self.rod_tip_sphere.translate(rod_pos, relative=False)
                    self.rod_tip_axes.translate(rod_pos, relative=False)
                    vis.update_geometry(self.rod_tip_sphere)
                    vis.update_geometry(self.rod_tip_axes)

                if to_reset:
                    vis.reset_view_point(True)
                    to_reset = False

                vis.poll_events()
                vis.update_renderer()

            except queue.Empty:
                vis.poll_events()
                vis.update_renderer()
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
                break

        vis.destroy_window()
        print("Visualization thread stopped.")

    def run(self, camera):
        """运行整合系统"""
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualizer_thread, daemon=True)
        self.vis_thread.start()

        intrinsics = camera.get_intrinsics()
        frame_count = 0
        last_fps_time = time.time()
        fps = 0

        try:
            while self.running:
                color_frame, depth_frame = camera.get_frame()
                if color_frame is None or depth_frame is None:
                    continue

                # 创建显示图像
                display_image = color_frame.copy()

                # ===== 1. 检测圆孔 =====
                circle = self.circle_detector.detect(color_frame)
                circle_center_3d = None
                pcd_circle = None

                if circle is not None:
                    cx, cy, r = map(int, circle)
                    cv2.circle(display_image, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(display_image, (cx, cy), 2, (0, 0, 255), 3)

                    # 计算圆心3D坐标
                    circle_center_3d = self.circle_extractor.get_circle_center_3d(
                        circle, depth_frame, camera
                    )

                    if circle_center_3d is not None:
                        text = f"Circle: ({circle_center_3d[0]:.3f}, {circle_center_3d[1]:.3f}, {circle_center_3d[2]:.3f})m"
                        cv2.putText(display_image, text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.putText(display_image, f"Circle ({cx},{cy}) r={r}",
                                (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                    # 提取点云
                    pcd_circle = self.circle_extractor.extract(
                        None, circle, color_frame, depth_frame, intrinsics, camera
                    )

                # ===== 2. 检测AprilTag =====
                corners, tag_id = self.apriltag_estimator.detect_tag(color_frame)
                rod_position = None

                if corners is not None:
                    rvec, tvec = self.apriltag_estimator.estimate_pose(corners)

                    if rvec is not None:
                        rod_position, rod_rvec = self.apriltag_estimator.compute_rod_tip_6d(rvec, tvec)

                        # 绘制AprilTag
                        display_image = self.apriltag_estimator.draw_on_image(
                            display_image, corners, rvec, tvec, rod_position, rod_rvec
                        )

                        # 显示6D位姿信息
                        R_rod, _ = cv2.Rodrigues(rod_rvec)
                        roll, pitch, yaw = self.apriltag_estimator.rotation_matrix_to_euler(R_rod)

                        pos = rod_position.reshape(-1)
                        text_rod = f"Rod: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m"
                        cv2.putText(display_image, text_rod, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        text_euler = f"RPY: ({roll:.1f}, {pitch:.1f}, {yaw:.1f})deg"
                        cv2.putText(display_image, text_euler, (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # ===== 3. 更新3D可视化 =====
                if pcd_circle is not None and len(pcd_circle.points) > 0:
                    if self.pcd_queue.full():
                        try:
                            self.pcd_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.pcd_queue.put((pcd_circle, circle_center_3d, rod_position))

                # ===== 4. 计算FPS =====
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - last_fps_time)
                    last_fps_time = current_time

                cv2.putText(display_image, f"FPS: {fps:.1f}", (10, display_image.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # 显示2D检测结果
                cv2.imshow("Integrated Detection (Circle + AprilTag)", display_image)

                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

        finally:
            self.running = False
            self.pcd_queue.put(None)
            if self.vis_thread:
                self.vis_thread.join(timeout=2.0)
            cv2.destroyAllWindows()
            print("Integrated viewer exited.")


def create_rod_rotation_matrix(axis='y', angle_deg=0):
    """创建旋转矩阵"""
    angle_rad = np.radians(angle_deg)

    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        R = np.eye(3)

    return R


def main():
    """主函数：整合圆孔检测 + AprilTag 6D位姿估计"""

    # 初始化相机（共享）
    cap = RealSenseCamera()

    # 初始化圆孔检测
    circle_detector = CircleDetector()
    circle_extractor = CirclePointCloudExtractor(margin=10)

    # 初始化AprilTag检测
    TAG_SIZE = 0.007  # 7mm
    ROD_OFFSET = np.array([0.0, -0.0035, 0.0])  # 32mm in Y-axis
    ROD_ROTATION = None  # 或 create_rod_rotation_matrix('x', 90)

    apriltag_estimator = AprilTagPoseEstimator(
        camera=cap,
        tag_size=TAG_SIZE,
        rod_offset=ROD_OFFSET,
        rod_rotation=ROD_ROTATION
    )

    # 创建整合的可视化器
    viewer = IntegratedViewer(circle_extractor, circle_detector, apriltag_estimator)

    print("=" * 70)
    print("🎯 整合检测系统：圆孔 + AprilTag 6D位姿估计")
    print("=" * 70)
    print("2D窗口:")
    print("  - 绿色圆 = 检测到的圆孔")
    print("  - 绿色框 + 坐标轴 = AprilTag标签")
    print("  - 青色点 = 棒子末端")
    print()
    print("3D窗口:")
    print("  - 红色球体 + 坐标轴 = 圆心位置")
    print("  - 青色球体 + 坐标轴 = 棒子末端位置")
    print("  - 彩色点云 = 圆孔区域")
    print()
    print("AprilTag坐标系:")
    print("  - X轴(红): 角点0→1")
    print("  - Y轴(绿): 角点0→3")
    print("  - Z轴(蓝): 垂直标签")
    print()
    print("按ESC键退出")
    print("=" * 70)

    try:
        viewer.run(cap)
    finally:
        cap.stop()

        # 打印最终结果
        print("\n" + "=" * 70)
        print("最终检测结果:")
        print("=" * 70)
        if circle_extractor.circle_center_3d is not None:
            c = circle_extractor.circle_center_3d
            print(f"圆心坐标: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) m")
        else:
            print("圆心坐标: 未检测到")
        print("=" * 70)


if __name__ == "__main__":
    main()