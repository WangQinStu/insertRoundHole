import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import threading
import queue
import datetime


class RealSenseCamera:
    """
    RealSense 相机管理类：
    - 获取彩色/深度流
    - 获取点云
    - 可视化 RGB / Depth / RGB+Depth / PointCloud
    """

    def __init__(self, width=640, height=480, fps=30):
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
        print(f"Depth Scale: {self.depth_scale}")

        # 相机内参
        depth_stream = self.profile.get_stream(rs.stream.depth)
        depth_intr = depth_stream.as_video_stream_profile().get_intrinsics()

        print('=' * 55)
        print('相机内参')
        print('=' * 55)
        print(f'fx: {depth_intr.fx}')
        print(f'fy: {depth_intr.fy}')
        print(f'cx: {depth_intr.ppx}')
        print(f'cy: {depth_intr.ppy}')
        print(f'width: {depth_intr.width}, height: {depth_intr.height}')
        print('=' * 55)

        # 保存内参用于点云生成
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_intr.width,
            depth_intr.height,
            depth_intr.fx,
            depth_intr.fy,
            depth_intr.ppx,
            depth_intr.ppy
        )

    def _pcd_visualizer_thread(self):
        """
        独立线程：Open3D 点云实时可视化
        """
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("RealSense d405 PointCloud Stream", width=800, height=600)

        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        # 保存点云回调函数
        def save_pcd_callback(vis_obj):
            if not pcd.is_empty():
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pcd_{now}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                print(f"✓ Saved point cloud to {filename}")
            return False

        # 注册 S 键保存点云
        vis.register_key_callback(ord("S"), save_pcd_callback)

        # 设置默认视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])

        to_reset = True

        print("Point cloud visualization started. Press 'S' to save current point cloud.")

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
                continue
            except Exception as e:
                print(f"Visualization error: {e}")
                break

        vis.destroy_window()
        print("Visualization thread stopped.")

    # -----------------------------------------------------------------------
    # 获取帧
    # -----------------------------------------------------------------------
    def get_frame(self):
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

    # -----------------------------------------------------------------------
    # 彩色流
    # -----------------------------------------------------------------------
    def get_color_stream(self):
        """获取彩色图像"""
        color, _ = self.get_frame()
        return color

    # -----------------------------------------------------------------------
    # 深度流
    # -----------------------------------------------------------------------
    def get_depth_stream(self):
        """获取深度图像"""
        _, depth = self.get_frame()
        return depth

    # -----------------------------------------------------------------------
    # 点云生成（使用 Open3D RGBD 方法）
    # -----------------------------------------------------------------------
    def get_point_cloud_o3d(self):
        """
        使用 Open3D RGBD 方法生成点云（更稳定）
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None

            # 可选：应用空间滤波器提高点云质量
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
                self.intrinsics
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
    # 可视化 RGB
    # -----------------------------------------------------------------------
    def visualize_rgb(self):
        """可视化RGB流"""
        print("Visualizing RGB stream. Press ESC to exit.")
        while True:
            color = self.get_color_stream()
            if color is None:
                continue

            cv2.imshow("RGB Stream", color)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # 可视化 Depth
    # -----------------------------------------------------------------------
    def visualize_depth(self):
        """可视化深度流"""
        print("Visualizing Depth stream. Press ESC to exit.")
        while True:
            depth = self.get_depth_stream()
            if depth is None:
                continue

            # 深度可视化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            cv2.imshow("Depth Stream", depth_colormap)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cv2.destroyAllWindows()

    # -----------------------------------------------------------------------
    # RGB + Depth 组合展示
    # -----------------------------------------------------------------------
    def visualize_rgb_depth(self):
        """同时可视化RGB和深度流"""
        print("Visualizing RGB + Depth. Press ESC to exit.")
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

    # -----------------------------------------------------------------------
    # 可视化点云 (Open3D + 多线程)
    # -----------------------------------------------------------------------
    def visualize_point_cloud(self):
        """实时可视化点云（多线程）"""
        print("Starting point cloud visualization. Press Ctrl+C to exit.")

        self.running = True

        # 启动可视化线程
        if self.vis_thread is None:
            self.vis_thread = threading.Thread(
                target=self._pcd_visualizer_thread, daemon=True
            )
            self.vis_thread.start()

        try:
            frame_count = 0
            while self.running:
                # 生成点云
                pcd = self.get_point_cloud_o3d()

                if pcd is None or len(pcd.points) == 0:
                    print("Warning: Empty point cloud, skipping...")
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
                    print(f"Processed {frame_count} frames, points: {len(pcd.points)}")

        except KeyboardInterrupt:
            print("\nStopping point cloud visualization...")

        finally:
            self.running = False
            self.pcd_queue.put(None)  # 发送终止信号
            if self.vis_thread:
                self.vis_thread.join(timeout=2.0)

    # -----------------------------------------------------------------------
    def stop(self):
        """停止相机"""
        self.running = False
        try:
            self.pipeline.stop()
            print("Camera stopped successfully.")
        except Exception as e:
            print(f"Error stopping camera: {e}")

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop()


# -----------------------------------------------------------------------
# 示例用法
# -----------------------------------------------------------------------
if __name__ == '__main__':
    with RealSenseCamera(width=640, height=480, fps=30) as camera:
        # 选择可视化模式
        print("\n=== RealSense Camera Visualization ===")
        print("1: RGB Stream")
        print("2: Depth Stream")
        print("3: RGB + Depth")
        print("4: Point Cloud (Press 'S' to save)")

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
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()