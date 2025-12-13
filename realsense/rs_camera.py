# 相机控制
import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import threading
import queue
import datetime
from typing import Tuple, Optional


class RealSenseCamera:
    """
    RealSense 相机管理类：
    - 获取彩色/深度流
    - 获取点云
    - 获取相机内参和畸变系数
    - 可视化 RGB / Depth / RGB+Depth / PointCloud
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pcd_queue = queue.Queue(maxsize=2)
        self.vis_thread = None
        self.running = False

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # depth 对齐到 color
        self.align = rs.align(rs.stream.color)

        # start streaming
        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            print(f"Failed to start pipeline: {e}")
            raise

        # 获取深度传感器参数
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 获取相机内参（深度流和彩色流）
        self._init_intrinsics()

        # 打印相机信息
        self._print_camera_info()

    def _init_intrinsics(self):
        """初始化相机内参和畸变系数"""
        # 深度流内参
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        # 彩色流内参
        color_stream = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Open3D 内参（用于点云生成，使用深度流内参）
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            self.depth_intrinsics.width,
            self.depth_intrinsics.height,
            self.depth_intrinsics.fx,
            self.depth_intrinsics.fy,
            self.depth_intrinsics.ppx,
            self.depth_intrinsics.ppy
        )

        # OpenCV 格式的相机内参矩阵（使用彩色流内参）
        self.camera_matrix = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 畸变系数（彩色流）
        coeffs = self.color_intrinsics.coeffs
        self.distortion_coeffs = np.array([
            coeffs[0],  # k1 - 径向畸变
            coeffs[1],  # k2 - 径向畸变
            coeffs[2],  # p1 - 切向畸变
            coeffs[3],  # p2 - 切向畸变
            coeffs[4]  # k3 - 径向畸变
        ], dtype=np.float32)

    def _print_camera_info(self):
        """打印相机信息"""
        print('=' * 60)
        print('RealSense 相机信息')
        print('=' * 60)
        print(f'分辨率: {self.width}x{self.height} @ {self.fps}fps')
        print(f'深度缩放因子: {self.depth_scale:.6f}')
        print()

        print('--- 彩色流内参 ---')
        print(f'fx: {self.color_intrinsics.fx:.2f}')
        print(f'fy: {self.color_intrinsics.fy:.2f}')
        print(f'cx: {self.color_intrinsics.ppx:.2f}')
        print(f'cy: {self.color_intrinsics.ppy:.2f}')
        print()

        print('--- 畸变系数 ---')
        print(f'k1 (径向): {self.distortion_coeffs[0]:.6f}')
        print(f'k2 (径向): {self.distortion_coeffs[1]:.6f}')
        print(f'p1 (切向): {self.distortion_coeffs[2]:.6f}')
        print(f'p2 (切向): {self.distortion_coeffs[3]:.6f}')
        print(f'k3 (径向): {self.distortion_coeffs[4]:.6f}')
        print('=' * 60)

    # -----------------------------------------------------------------------
    # 获取相机参数的公共接口
    # -----------------------------------------------------------------------
    def get_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        """获取 Open3D 格式的相机内参"""
        return self.intrinsics_o3d

    def get_camera_matrix(self) -> np.ndarray:
        """获取 OpenCV 格式的相机内参矩阵 (3x3)"""
        return self.camera_matrix.copy()

    def get_distortion_coeffs(self) -> np.ndarray:
        """获取畸变系数 [k1, k2, p1, p2, k3]"""
        return self.distortion_coeffs.copy()

    def get_depth_scale(self) -> float:
        """获取深度缩放因子"""
        return self.depth_scale

    # -----------------------------------------------------------------------
    # 获取帧
    # -----------------------------------------------------------------------
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取对齐后的彩色和深度帧"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())

            return color_np, depth_np
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None, None

    def get_color_stream(self) -> Optional[np.ndarray]:
        """获取彩色图像"""
        color, _ = self.get_frame()
        return color

    def get_depth_stream(self) -> Optional[np.ndarray]:
        """获取深度图像"""
        _, depth = self.get_frame()
        return depth

    # -----------------------------------------------------------------------
    # 点云生成（使用 Open3D RGBD 方法）
    # -----------------------------------------------------------------------
    def get_point_cloud_o3d(self, apply_filter=True) -> Optional[o3d.geometry.PointCloud]:
        """
        使用 Open3D RGBD 方法生成点云

        Args:
            apply_filter: 是否应用空间滤波器提高点云质量

        Returns:
            点云对象或 None
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            # 可选：应用空间滤波器
            if apply_filter:
                spatial = rs.spatial_filter()
                spatial.set_option(rs.option.filter_magnitude, 2)
                spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
                spatial.set_option(rs.option.filter_smooth_delta, 20)
                depth_frame = spatial.process(depth_frame)

            # 转换为 numpy 数组
            depth = np.asanyarray(depth_frame.get_data())
            color = np.asanyarray(color_frame.get_data())

            # 创建 Open3D 图像
            o3d_depth = o3d.geometry.Image(depth)
            o3d_color = o3d.geometry.Image(color)

            # 从 RGBD 创建点云
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / self.depth_scale,
                depth_trunc=2.5,  # 截断距离 2.5米
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsics_o3d
            )

            # 修正坐标系（防止倒置）
            pcd.transform([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

            return pcd

        except Exception as e:
            print(f"Error getting point cloud: {e}")
            return None

    # -----------------------------------------------------------------------
    # 点云可视化线程
    # -----------------------------------------------------------------------
    def _pcd_visualizer_thread(self):
        """独立线程：Open3D 点云实时可视化"""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("RealSense PointCloud Stream", width=800, height=600)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        opt.point_size = 1.5

        # 添加坐标轴
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(axis)

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 保存点云回调
        def save_pcd_callback(vis_obj):
            if not pcd.is_empty():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pcd_{timestamp}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                print(f"✓ 点云已保存: {filename}")
            return False

        vis.register_key_callback(ord("S"), save_pcd_callback)

        # 设置默认视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.4)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])

        to_reset = True
        print("点云可视化已启动。按 'S' 键保存当前点云。")

        while self.running:
            try:
                new_pcd = self.pcd_queue.get(timeout=0.1)

                if new_pcd is None:  # 终止信号
                    break

                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                vis.update_geometry(pcd)

                if to_reset:
                    vis.reset_view_point(True)
                    to_reset = False

                vis.poll_events()
                vis.update_renderer()

            except queue.Empty:
                vis.poll_events()
                vis.update_renderer()
            except Exception as e:
                print(f"可视化错误: {e}")
                break

        vis.destroy_window()
        print("可视化线程已停止。")

    # -----------------------------------------------------------------------
    # 可视化方法
    # -----------------------------------------------------------------------
    def visualize_rgb(self):
        """可视化 RGB 流"""
        print("可视化 RGB 流。按 ESC 退出。")
        while True:
            color = self.get_color_stream()
            if color is None:
                continue

            cv2.imshow("RGB Stream", color)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cv2.destroyAllWindows()

    def visualize_depth(self):
        """可视化深度流"""
        print("可视化深度流。按 ESC 退出。")
        while True:
            depth = self.get_depth_stream()
            if depth is None:
                continue

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            cv2.imshow("Depth Stream", depth_colormap)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cv2.destroyAllWindows()

    def visualize_rgb_depth(self):
        """同时可视化 RGB 和深度流"""
        print("可视化 RGB + 深度。按 ESC 退出。")
        while True:
            color, depth = self.get_frame()
            if color is None:
                continue

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            combined = np.hstack((color, depth_colormap))
            cv2.imshow("RGB + Depth", combined)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cv2.destroyAllWindows()

    def visualize_point_cloud(self):
        """实时可视化点云（多线程）"""
        print("启动点云可视化。按 Ctrl+C 退出。")

        self.running = True

        # 启动可视化线程
        if self.vis_thread is None or not self.vis_thread.is_alive():
            self.vis_thread = threading.Thread(
                target=self._pcd_visualizer_thread, daemon=True
            )
            self.vis_thread.start()

        try:
            frame_count = 0
            while self.running:
                pcd = self.get_point_cloud_o3d()

                if pcd is None or len(pcd.points) == 0:
                    continue

                # 放入队列（丢弃旧帧）
                if self.pcd_queue.full():
                    try:
                        self.pcd_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.pcd_queue.put(pcd)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧，点数: {len(pcd.points)}")

        except KeyboardInterrupt:
            print("\n停止点云可视化...")
        finally:
            self.running = False
            self.pcd_queue.put(None)  # 发送终止信号
            if self.vis_thread:
                self.vis_thread.join(timeout=2.0)

    # -----------------------------------------------------------------------
    # 资源管理
    # -----------------------------------------------------------------------
    def stop(self):
        """停止相机"""
        self.running = False
        try:
            self.pipeline.stop()
            print("相机已成功停止。")
        except Exception as e:
            print(f"停止相机时出错: {e}")

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()

    def __del__(self):
        """析构函数"""
        self.stop()


# -----------------------------------------------------------------------
# 示例用法
# -----------------------------------------------------------------------
if __name__ == '__main__':
    with RealSenseCamera(width=640, height=480, fps=30) as camera:
        # 打印相机参数
        print("\n获取相机参数示例:")
        K = camera.get_camera_matrix()
        dist = camera.get_distortion_coeffs()
        print(f"\n相机矩阵 K:\n{K}")
        print(f"\n畸变系数:\n{dist}")

        # 选择可视化模式
        print("\n=== RealSense 相机可视化 ===")
        print("1: RGB 流")
        print("2: 深度流")
        print("3: RGB + 深度")
        print("4: 点云（按 'S' 保存）")

        mode = input("\n选择模式 [1-4]: ").strip()

        try:
            if mode == '1':
                camera.visualize_rgb()
            elif mode == '2':
                camera.visualize_depth()
            elif mode == '3':
                camera.visualize_rgb_depth()
            elif mode == '4':
                camera.visualize_point_cloud()
            else:
                print("默认使用点云可视化")
                camera.visualize_point_cloud()

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"错误: {e}")
            import traceback

            traceback.print_exc()