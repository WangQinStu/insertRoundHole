import cv2
import numpy as np
from realsense.rs_camera import RealSenseCamera


class AprilTagPoseEstimator:
    """
    AprilTag 位姿估计 + 铁棒末端6D位姿计算
    """

    def __init__(self, camera, tag_size: float, rod_offset: np.ndarray, rod_rotation: np.ndarray = None):
        """
        Args:
            camera: RealSenseCamera 实例
            tag_size: 标签边长（米）
            rod_offset: 铁棒末端相对于标签中心的偏移 (3,1)，单位米
            rod_rotation: 铁棒相对于标签的旋转矩阵 (3,3)，默认为None（姿态与标签相同）
        """
        self.camera = camera
        self.tag_size = tag_size
        self.rod_offset = rod_offset.reshape(3, 1)

        # 棒子相对于标签的旋转矩阵
        if rod_rotation is None:
            self.rod_rotation = np.eye(3)  # 默认无旋转
        else:
            self.rod_rotation = rod_rotation

        # 相机内参
        self.K = camera.get_camera_matrix()
        self.dist = camera.get_distortion_coeffs()

        # 选择 AprilTag 字典
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # 标签三维角点（标签局部坐标系）???
        s = tag_size / 2.0
        self.obj_points = np.array([
            [-s, s, 0],  # 角点0：左上
            [s, s, 0],  # 角点1：右上
            [s, -s, 0],  # 角点2：右下
            [-s, -s, 0]  # 角点3：左下
        ], dtype=np.float32)

    def detect_tag(self, image):
        """
        返回:
            corners (4,2): 图像坐标系下的角点
            ids: 标签ID
        """
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is None:
            return None, None
        return corners[0].reshape(4, 2).astype(np.float32), int(ids[0][0])

    def estimate_pose(self, corners):
        """
        solvePnP 得到 rvec/tvec
        返回标签在相机坐标系下的位姿
        """
        ok, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return (rvec, tvec) if ok else (None, None)

    def compute_rod_tip_6d(self, rvec, tvec):
        """
        计算铁棒末端的6D位姿（位置 + 姿态）

        返回:
            rod_position: (3,1) 棒子末端在相机坐标系的位置
            rod_rvec: (3,1) 棒子末端在相机坐标系的姿态（旋转向量）
        """
        R_tag, _ = cv2.Rodrigues(rvec)  # 标签的旋转矩阵

        # 1. 计算棒子末端位置
        rod_position = tvec + R_tag @ self.rod_offset

        # 2. 计算棒子姿态
        # 棒子在相机坐标系的旋转 = 标签旋转 × 棒子相对标签的旋转
        R_rod = R_tag @ self.rod_rotation
        rod_rvec, _ = cv2.Rodrigues(R_rod)

        return rod_position, rod_rvec

    def rotation_matrix_to_euler(self, R):
        """
        旋转矩阵转欧拉角（XYZ顺规）
        返回: (roll, pitch, yaw) 单位：度
        """
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

    def draw(self, image, corners, rvec, tvec, rod_position, rod_rvec):
        img = image.copy()

        # 绘制标签框
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 3)

        # 绘制角点编号
        for i, c in enumerate(corners):
            x, y = int(c[0]), int(c[1])
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(img, str(i), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # 绘制标签坐标轴 (X=红, Y=绿, Z=蓝)
        cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, self.tag_size, 3)

        # ===== 铁棒末端投影 =====
        rod_tip_2d, _ = cv2.projectPoints(
            rod_position.reshape(1, 3),
            np.zeros((3, 1)),  # 零旋转（已在相机坐标系）
            np.zeros((3, 1)),  # 零平移
            self.K,
            self.dist
        )

        rod_tip_2d = tuple(rod_tip_2d[0][0].astype(int))
        cv2.circle(img, rod_tip_2d, 7, (0, 255, 0), -1)
        cv2.putText(
            img, "Rod Tip",
            (rod_tip_2d[0] + 10, rod_tip_2d[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2
        )

        pos = rod_position.reshape(-1)
        text = f"Rod XYZ: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"

        cv2.putText(
            img,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # 绘制棒子的坐标轴（棒子自己的姿态）
        axis_length = self.tag_size * 1.5
        cv2.drawFrameAxes(img, self.K, self.dist, rod_rvec, rod_position, axis_length, 2)

        # 添加标签中心点（用于对比）
        tag_center_2d, _ = cv2.projectPoints(
            np.zeros((1, 3)),
            rvec,
            tvec,
            self.K,
            self.dist
        )
        tag_center_2d = tuple(tag_center_2d[0][0].astype(int))
        cv2.circle(img, tag_center_2d, 5, (255, 255, 0), -1)

        return img


def create_rod_rotation_matrix(axis='y', angle_deg=0):
    """
    创建棒子相对于标签的旋转矩阵

    Args:
        axis: 旋转轴 ('x', 'y', 'z')
        angle_deg: 旋转角度（度）

    常见情况:
    - 棒子沿标签Y轴延伸，姿态与标签相同: 返回单位矩阵
    - 棒子沿Y轴但想让棒子的Z轴指向棒子方向: 绕X轴旋转90度
    """
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
    cap = RealSenseCamera()

    # ===== 参数设置 =====
    TAG_SIZE = 0.008 # 标签大小

    # 棒子末端相对标签中心的位置偏移
    ROD_OFFSET = np.array([-0.034, 0.0,  0.0])  # (X, Y, Z) 米

    # 棒子相对标签的旋转（可选）
    # 1: 姿态与标签完全相同
    ROD_ROTATION = None  # 或 np.eye(3)
    # 2: 如果想让棒子的Z轴指向棒子延伸方向
    # ROD_ROTATION = create_rod_rotation_matrix('x', 90)  # 绕X轴旋转90度
    # 3: 自定义旋转
    # ROD_ROTATION = create_rod_rotation_matrix('z', 45)  # 绕Z轴旋转45度

    estimator = AprilTagPoseEstimator(
        camera=cap,
        tag_size=TAG_SIZE,
        rod_offset=ROD_OFFSET,
        rod_rotation=ROD_ROTATION
    )

    print("AprilTag坐标系说明:")
    print("- X轴(红): 角点0→1方向")
    print("- Y轴(绿): 角点0→3方向")
    print("- Z轴(蓝): 垂直标签向外")
    print(f"棒子位置偏移: {ROD_OFFSET} 米")
    print(f"棒子姿态旋转: {'与标签相同' if ROD_ROTATION is None else '自定义旋转'}\n")

    while True:
        frame = cap.get_color_stream()
        if frame is None:
            continue

        corners, tag_id = estimator.detect_tag(frame)
        if corners is not None:
            rvec, tvec = estimator.estimate_pose(corners)

            if rvec is not None:
                # 获取棒子的6D位姿
                rod_position, rod_rvec = estimator.compute_rod_tip_6d(rvec, tvec)

                # 转换为欧拉角（方便理解）
                R_rod, _ = cv2.Rodrigues(rod_rvec)
                roll, pitch, yaw = estimator.rotation_matrix_to_euler(R_rod)

                # 可视化
                vis = estimator.draw(frame, corners, rvec, tvec, rod_position, rod_rvec)

                # 打印6D位姿
                print(f"===== Tag ID {tag_id} =====")
                print(f"Tag 位置: {tvec.reshape(-1)}")
                print(f"Rod 位置: {rod_position.reshape(-1)}")
                print(f"Rod 姿态 (rvec): {rod_rvec.reshape(-1)}")
                print(f"Rod 姿态 (欧拉角 RPY): roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°")
                print(f"距离: {np.linalg.norm(rod_position - tvec):.4f} m\n")
            else:
                vis = frame
        else:
            vis = frame

        cv2.imshow("AprilTag Rod 6D Pose Estimation", vis)
        if cv2.waitKey(1) == 27:
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()