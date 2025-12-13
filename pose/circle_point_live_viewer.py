import cv2
import open3d as o3d
import numpy as np
import threading
import queue
import time

class CirclePointCloudLiveViewer:
    def __init__(self, extractor, detector):
        """
        Args:
            extractor: CirclePointCloudExtractor实例
            detector: CircleDetector实例
        """
        self.extractor = extractor
        self.detector = detector

        # 点云队列
        self.pcd_queue = queue.Queue(maxsize=2)
        self.running = False
        self.vis_thread = None

        # 圆心小球
        self.center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        self.center_sphere.paint_uniform_color([1, 0, 0])  # 红色

        # 圆心文本标注（用三个坐标轴表示）
        self.center_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02,origin=[0, 0, 0])

    def _visualizer_thread(self):
        """Open3D可视化线程"""
        vis = o3d.visualization.Visualizer()
        vis.create_window("Circle Point Cloud Live", width=800, height=600)

        # 设置渲染参数
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        opt.point_size = 2.0

        # 添加坐标系
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coordinate)

        # 初始化空点云
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 添加圆心小球
        vis.add_geometry(self.center_sphere)

        # 添加坐标轴
        vis.add_geometry(self.center_axes)

        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])

        to_reset = True
        print("Point cloud visualization started.")

        while self.running:
            try:
                new_pcd = self.pcd_queue.get(timeout=0.1)
                if new_pcd is None:  # 终止信号
                    break

                # 更新点云
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors

                vis.update_geometry(pcd)

                # 更新圆心位置（关键修改！）
                if self.extractor.circle_center_3d is not None:
                    center = self.extractor.circle_center_3d
                    center = np.array([center[0], - center[1], - center[2]])
                    # 更新球体位置
                    current_center = np.asarray(self.center_sphere.get_center())
                    translation = center - current_center
                    self.center_sphere.translate(translation, relative=False)

                    # 更新坐标轴位置
                    self.center_axes.translate(center, relative=False)

                    # vis.update_geometry(self.center_sphere)
                    vis.update_geometry(self.center_axes)

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
        """运行实时可视化"""
        self.running = True
        self.vis_thread = threading.Thread(target=self._visualizer_thread, daemon=True)
        self.vis_thread.start()

        intrinsics = camera.get_intrinsics()
        frame_count = 0
        fps_time = time.time()

        try:
            while self.running:
                color_frame, depth_frame = camera.get_frame()
                if color_frame is None or depth_frame is None:
                    continue

                # 检测圆
                circle = self.detector.detect(color_frame)
                result = color_frame.copy()

                if circle is not None:
                    cx, cy, r = map(int, circle)
                    cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
                    cv2.circle(result, (cx, cy), 2, (0, 0, 255), 3)

                    # 计算圆心3D坐标（关键添加！）
                    center_3d = self.extractor.get_circle_center_3d(
                        circle, depth_frame, camera
                    )

                    # 在2D图像上显示3D坐标
                    if center_3d is not None:
                        text = f"Center: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})m"
                        cv2.putText(result, text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    cv2.putText(result, f"({cx},{cy}) r={r}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # 提取圆区域点云
                    pcd_circle = self.extractor.extract(None, circle, color_frame, depth_frame, intrinsics, camera)

                    # 放入队列更新显示
                    if pcd_circle is not None and len(pcd_circle.points) > 0:
                        if self.pcd_queue.full():
                            try:
                                self.pcd_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.pcd_queue.put(pcd_circle)

                # 显示2D检测结果
                cv2.imshow("Circle Detection 2D", result)

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
            print("Viewer exited.")
