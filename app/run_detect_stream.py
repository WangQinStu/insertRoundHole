from realsense.rs_camera import RealSenseCamera
from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor
from pose.circle_point_live_viewer import CirclePointCloudLiveViewer


def run_circle_detection():
    """Run the circle detection + 3D point cloud live viewer."""
    camera = RealSenseCamera()
    detector = CircleDetector()
    extractor = CirclePointCloudExtractor(margin=10)
    viewer = CirclePointCloudLiveViewer(extractor, detector)

    print("=" * 60)
    print("🎯 圆孔检测与3D定位系统")
    print("=" * 60)
    print("红色球体 = 圆心位置")
    print("彩色坐标轴 = 圆心坐标系")
    print("ESC键退出")
    print("=" * 60)

    try:
        viewer.run(camera)
    finally:
        camera.stop()

        if extractor.circle_center_3d is not None:
            print(f"\n最终圆心坐标: {extractor.circle_center_3d}")


if __name__ == "__main__":
    run_circle_detection()
