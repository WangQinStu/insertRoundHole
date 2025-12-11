from realsense.rs_camera import RealSenseCamera
from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor
from pose.circle_point_live_viewer import CirclePointCloudLiveViewer

def main():
    # 初始化相机
    cap = RealSenseCamera()

    # 初始化检测器和提取器
    detector = CircleDetector()
    extractor = CirclePointCloudExtractor(margin=10)


    # 创建可视化器并运行
    viewer = CirclePointCloudLiveViewer(extractor, detector)
    print('圆心：', viewer.center_sphere)
    try:
        viewer.run(cap)
    finally:
        cap.stop()

if __name__ == "__main__":
    main()
