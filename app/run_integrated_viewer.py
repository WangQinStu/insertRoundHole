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


def align_model_z_to_normal(normal: np.ndarray) -> np.ndarray:
    """Return a rotation that aligns +Z to the given normal vector."""
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, normal)
    if s < 1e-6:
        return np.eye(3) if c > 0 else np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a noisy rotation back to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def project_normal_to_image(center_cam: np.ndarray, normal_cam: np.ndarray, K: np.ndarray, dist: np.ndarray, scale: float = 0.05):
    """Project a 3D normal arrow to pixel coordinates for visualization."""
    try:
        p0 = center_cam.reshape(1, 1, 3)
        p1 = (center_cam + normal_cam * scale).reshape(1, 1, 3)
        proj0, _ = cv2.projectPoints(p0, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
        proj1, _ = cv2.projectPoints(p1, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
        return tuple(proj0[0, 0].astype(int)), tuple(proj1[0, 0].astype(int))
    except Exception:
        return None, None


def estimate_normal_multiscale(circle, depth_image, intrinsics_o3d, depth_scale):
    """Multiscale ring sampling + RANSAC plane fit for a stable hole normal."""
    if circle is None:
        return None, None

    cx, cy, r = circle
    fx = intrinsics_o3d.intrinsic_matrix[0, 0]
    fy = intrinsics_o3d.intrinsic_matrix[1, 1]
    cx_intr = intrinsics_o3d.intrinsic_matrix[0, 2]
    cy_intr = intrinsics_o3d.intrinsic_matrix[1, 2]
    h, w = depth_image.shape

    depth_filtered = cv2.bilateralFilter(depth_image.astype(np.float32), d=5, sigmaColor=10, sigmaSpace=10).astype(depth_image.dtype)
    scales = [(0.4, 0.8), (0.8, 1.3), (1.3, 1.8)]

    all_pts = []
    scale_counts = []
    for inner_ratio, outer_ratio in scales:
        max_r = int(r * outer_ratio)
        min_r = int(r * inner_ratio)
        ys, xs = np.ogrid[0:h, 0:w]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        ring_mask = (dist >= min_r) & (dist <= max_r)
        ys_idx, xs_idx = np.where(ring_mask)
        if len(xs_idx) < 20:
            scale_counts.append(0)
            continue

        depths = depth_filtered[ys_idx, xs_idx]
        valid_mask = depths > 0
        if np.count_nonzero(valid_mask) < 20:
            scale_counts.append(0)
            continue

        depths = depths[valid_mask] * depth_scale
        xs_idx = xs_idx[valid_mask]
        ys_idx = ys_idx[valid_mask]

        zs = depths
        xs_cam = (xs_idx - cx_intr) * zs / fx
        ys_cam = (ys_idx - cy_intr) * zs / fy
        pts = np.stack([xs_cam, ys_cam, zs], axis=1)

        if len(pts) > 50:
            median_z = np.median(pts[:, 2])
            mad = np.median(np.abs(pts[:, 2] - median_z))
            if mad > 1e-6:
                threshold = 3 * mad * 1.4826
                pts = pts[np.abs(pts[:, 2] - median_z) < threshold]

        scale_counts.append(len(pts))
        all_pts.append(pts)

    if len(all_pts) == 0:
        return None, None

    all_pts = np.vstack(all_pts)
    if len(all_pts) < 50:
        return None, None

    best_normal = None
    best_inliers = 0
    ransac_iterations = 80
    inlier_threshold = 0.002
    for _ in range(ransac_iterations):
        idx = np.random.choice(len(all_pts), 3, replace=False)
        p1, p2, p3 = all_pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        d = np.abs((all_pts - p1) @ normal)
        inliers = np.sum(d < inlier_threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal

    if best_normal is None or best_inliers < 50:
        pts_centered = all_pts - all_pts.mean(axis=0)
        cov = pts_centered.T @ pts_centered / len(pts_centered)
        _, _, vh = np.linalg.svd(cov)
        best_normal = vh[-1, :]
        best_inliers = len(all_pts)

    if best_normal[2] < 0:
        best_normal = -best_normal

    stats = {
        "total_points": len(all_pts),
        "inliers": best_inliers,
        "inlier_ratio": best_inliers / len(all_pts),
        "scale_counts": scale_counts,
        "median_depth": float(np.median(all_pts[:, 2])),
        "depth_std": float(np.std(all_pts[:, 2])),
    }
    return best_normal, stats


def run_integrated_detection(camera: RealSenseCamera | None = None):
    """
    Run circle detector and AprilTag-based 6D pose estimation in a single viewer.

    If a camera instance is provided it will be reused, so circle detection and AprilTag
    pose estimation share a single RealSense stream without grabbing the device twice.
    """
    owns_camera = camera is None
    camera = camera or RealSenseCamera()
    circle_detector = CircleDetector()
    circle_extractor = CirclePointCloudExtractor(margin=10)
    last_circle = None
    last_circle_3d = None

    tag_size = 0.007  # meters
    rod_offset = np.array([-0.0035, 0.0,  -0.02])  # meters
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
    last_circle_normal = None
    last_circle_R = None
    last_circle_rpy = None
    alpha_normal = 0.2
    intrinsics_o3d = camera.get_intrinsics()
    depth_scale = camera.depth_scale

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
            circle_pose_text = "Circle RPY (deg): ---, ---, ---"

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

            if circle_center_3d is not None:
                normal, _ = estimate_normal_multiscale(circle, depth_frame, intrinsics_o3d, depth_scale)
                if normal is not None:
                    if last_circle_normal is not None and np.dot(normal, last_circle_normal) < 0:
                        normal = -normal
                    if last_circle_normal is not None:
                        normal = last_circle_normal * (1 - alpha_normal) + normal * alpha_normal
                        normal = normal / (np.linalg.norm(normal) + 1e-8)
                    last_circle_normal = normal

                    R_circle = align_model_z_to_normal(normal)
                    if last_circle_R is not None:
                        R_blend = last_circle_R * (1 - alpha_pose) + R_circle * alpha_pose
                        R_circle = orthonormalize(R_blend)
                    last_circle_R = R_circle

                    circle_rpy = pose_estimator.rotation_matrix_to_euler(R_circle)
                    last_circle_rpy = circle_rpy
                    circle_pose_text = f"Circle RPY (deg): {circle_rpy[0]:.1f}, {circle_rpy[1]:.1f}, {circle_rpy[2]:.1f}"

                    start, end = project_normal_to_image(circle_center_3d, normal, pose_estimator.K, pose_estimator.dist, scale=0.05)
                    if start and end:
                        cv2.arrowedLine(display, start, end, (0, 255, 255), 2, tipLength=0.2)
                        cv2.putText(display, "Circle Normal", (end[0] + 5, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            elif last_circle_rpy is not None:
                circle_pose_text = f"Circle RPY (deg): {last_circle_rpy[0]:.1f}, {last_circle_rpy[1]:.1f}, {last_circle_rpy[2]:.1f}"

            cv2.putText(display, rod_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, rpy_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, circle_pose_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Circle + AprilTag Viewer", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if owns_camera:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_integrated_detection()
