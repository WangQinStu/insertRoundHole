import cv2
import numpy as np
from realsense.rs_camera import RealSenseCamera


class AprilTagPoseEstimator:
    """
    AprilTag 位姿估计 + 铁棒末端位置计算
    """

    def __init__(self, camera, tag_size: float, rod_offset: np.ndarray):
        """
        Args:
            camera: RealSenseCamera 实例
            tag_size: 标签边长（米）
            rod_offset: 铁棒末端相对于标签中心的偏移 (3,1)，单位米
        """
        self.camera = camera
        self.tag_size = tag_size
        self.rod_offset = rod_offset.reshape(3, 1)

        # 相机内参
        self.K = camera.get_camera_matrix()
        self.dist = camera.get_distortion_coeffs()

        # 选择 AprilTag 字典
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # 标签三维角点（世界坐标）
        s = tag_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

    def detect_tag(self, image):
        """
        返回:
            corners (4,2)
            ids
        """
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            return None, None
        return corners[0].reshape(4, 2).astype(np.float32), int(ids[0][0])

    def estimate_pose(self, corners):
        """
        solvePnP 得到 rvec/tvec
        """
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if ok else (None, None)

    def compute_rod_tip(self, rvec, tvec):
        """
        根据标签姿态 + 几何偏移 计算铁棒末端位置
        """
        R, _ = cv2.Rodrigues(rvec)
        rod_tip_tvec = tvec + R @ self.rod_offset
        return rod_tip_tvec, rvec  # 姿态同标签

    def draw(self, image, corners, rvec, tvec, rod_tip):
        """
        可视化：边框、坐标轴、铁棒末端点等
        """
        img = image.copy()

        # 绘制标签框
        pts = corners.astype(np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 3)

        # 绘制角点
        for i, c in enumerate(corners):
            x, y = int(c[0]), int(c[1])

            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # 角点
            cv2.putText(img, str(i), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # 标签坐标轴
        cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, self.tag_size, 3)

        # --- 绘制铁棒末端 ---
        # rod_tip_tvec 是 3x1，需要 reshape 成 (1,3)
        rod_tip_point = rod_tip_tvec.reshape(1, 3).astype(np.float32)

        rod_tip_2d, _ = cv2.projectPoints(
            rod_tip_point,
            rvec,
            tvec,
            self.K,
            self.dist
        )

        rod_tip_2d = tuple(rod_tip_2d[0][0].astype(int))
        cv2.circle(img, rod_tip_2d, 7, (0, 255, 255), -1)
        cv2.putText(
            img, "Rod End",
            (rod_tip_2d[0], rod_tip_2d[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2
        )


def main():
    cap = RealSenseCamera()

    # ===== 必须手工测量 =====
    TAG_SIZE = 0.02       # 标签边长 20mm
    ROD_OFFSET = np.array([0.0, 0.0, 0.12])  # 标签 -> 铁棒末端 120mm（示例）

    estimator = AprilTagPoseEstimator(
        camera=cap,
        tag_size=TAG_SIZE,
        rod_offset=ROD_OFFSET
    )

    while True:
        frame = cap.get_color_stream()
        if frame is None:
            continue

        corners, tag_id = estimator.detect_tag(frame)
        if corners is not None:
            rvec, tvec = estimator.estimate_pose(corners)

            if rvec is not None:
                rod_tip_tvec, rod_tip_rvec = estimator.compute_rod_tip(rvec, tvec)

                # 可视化
                vis = estimator.draw(frame, corners, rvec, tvec, rod_tip_tvec)

                # 打印坐标
                print(f"Tag ID {tag_id}: {tvec.reshape(-1)} (m)")
                print(f"Rod End: {rod_tip_tvec.reshape(-1)} (m)")
            else:
                vis = frame
        else:
            vis = frame

        cv2.imshow("AprilTag Rod Pose Estimation", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
