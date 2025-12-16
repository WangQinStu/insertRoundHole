import cv2
import numpy as np

from detection.QRCodeDetector import AprilTagPoseEstimator
from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor
from realsense.rs_camera import RealSenseCamera


def draw_apriltag_overlay(image, corners, rvec, tvec, camera_matrix, dist_coeffs):
    """Draws AprilTag outline, corner indices, and pose axes."""
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 0), 2)

    for idx, corner in enumerate(corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.02, 2)
    return image


def run_integrated_detection():
    """Run circle detector and AprilTag-based 6D pose estimation in a single viewer."""
    camera = RealSenseCamera()
    circle_detector = CircleDetector()
    circle_extractor = CirclePointCloudExtractor(margin=10)
    last_circle = None
    last_circle_3d = None

    tag_size = 0.007  # meters
    rod_offset = np.array([0.0, 0.032, 0.0])  # meters
    rod_rotation = None

    pose_estimator = AprilTagPoseEstimator(
        camera=camera,
        tag_size=tag_size,
        rod_offset=rod_offset,
        rod_rotation=rod_rotation,
    )

    print("=" * 70)
    print("Integrated viewer: Circle detector + AprilTag 6D pose")
    print("ESC to exit")
    print("=" * 70)

    last_rod_position = None
    last_rpy = None
    alpha_pose = 0.25  # smoothing factor for rod pose

    try:
        while True:
            color_frame, depth_frame = camera.get_frame()
            if color_frame is None or depth_frame is None:
                continue

            display = color_frame.copy()

            circle = circle_detector.detect(color_frame)
            circle_center_3d = None

            if circle is not None:
                last_circle = circle
                cx, cy, r = map(int, circle)
                cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), 3)
                cv2.putText(display, f"Circle ({cx},{cy}) r={r}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                circle_center_3d = circle_extractor.get_circle_center_3d(circle, depth_frame, camera)
                if circle_center_3d is not None:
                    last_circle_3d = circle_center_3d
                    text = f"Circle XYZ (m): {circle_center_3d[0]:.3f}, {circle_center_3d[1]:.3f}, {circle_center_3d[2]:.3f}"
                    cv2.putText(display, text, (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            elif last_circle is not None:
                cx, cy, r = map(int, last_circle)
                cv2.circle(display, (cx, cy), r, (0, 180, 0), 1)
                cv2.circle(display, (cx, cy), 2, (0, 120, 120), 2)
                cv2.putText(display, "Circle (prev)", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 255), 1)

                if last_circle_3d is not None:
                    text = f"Circle XYZ (m): {last_circle_3d[0]:.3f}, {last_circle_3d[1]:.3f}, {last_circle_3d[2]:.3f}"
                    cv2.putText(display, text, (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            rod_text = "Rod XYZ (m): ---, ---, ---"
            rpy_text = "Rod RPY (deg): ---, ---, ---"

            corners, tag_id = pose_estimator.detect_tag(color_frame)
            if corners is not None:
                rvec, tvec = pose_estimator.estimate_pose(corners)
                if rvec is not None and tvec is not None:
                    rod_position, rod_rvec = pose_estimator.compute_rod_tip_6d(rvec, tvec)

                    display = draw_apriltag_overlay(display, corners, rvec, tvec, pose_estimator.K, pose_estimator.dist)

                    rod_R, _ = cv2.Rodrigues(rod_rvec)
                    roll, pitch, yaw = pose_estimator.rotation_matrix_to_euler(rod_R)

                    if last_rod_position is not None:
                        rod_position = last_rod_position * (1 - alpha_pose) + rod_position * alpha_pose
                    if last_rpy is not None:
                        roll = last_rpy[0] * (1 - alpha_pose) + roll * alpha_pose
                        pitch = last_rpy[1] * (1 - alpha_pose) + pitch * alpha_pose
                        yaw = last_rpy[2] * (1 - alpha_pose) + yaw * alpha_pose

                    last_rod_position = rod_position
                    last_rpy = np.array([roll, pitch, yaw])

                    pos = rod_position.reshape(-1)
                    rod_text = f"Rod XYZ (m): {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
                    rpy_text = f"Rod RPY (deg): {roll:.1f}, {pitch:.1f}, {yaw:.1f}"

                    # Project rod tip for a small visual marker
                    rod_tip_2d, _ = cv2.projectPoints(rod_position.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), pose_estimator.K, pose_estimator.dist)
                    rod_tip_pt = tuple(rod_tip_2d[0][0].astype(int))
                    cv2.circle(display, rod_tip_pt, 6, (0, 255, 255), -1)
                    cv2.putText(display, "Rod Tip", (rod_tip_pt[0] + 8, rod_tip_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            elif last_rod_position is not None and last_rpy is not None:
                pos = last_rod_position.reshape(-1)
                rod_text = f"Rod XYZ (m): {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}"
                rpy_text = f"Rod RPY (deg): {last_rpy[0]:.1f}, {last_rpy[1]:.1f}, {last_rpy[2]:.1f}"

            cv2.putText(display, rod_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, rpy_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Circle + AprilTag Viewer", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_integrated_detection()
