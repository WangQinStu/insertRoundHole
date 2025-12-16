import cv2
import numpy as np
import open3d as o3d

from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor
from pose.hole_icp import HoleICPRegister
from realsense.rs_camera import RealSenseCamera


def project_normal_to_image(center_cam: np.ndarray, normal_cam: np.ndarray, K: np.ndarray, dist: np.ndarray,
                            scale: float = 0.05):
    """
    将相机坐标系下的圆心与法向投影到图像平面，返回起点/终点像素坐标。
    scale: 法向箭头长度（米）
    """
    try:
        p0 = center_cam.reshape(1, 1, 3)
        p1 = (center_cam + normal_cam * scale).reshape(1, 1, 3)
        proj0, _ = cv2.projectPoints(p0, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
        proj1, _ = cv2.projectPoints(p1, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
        start = tuple(proj0[0, 0].astype(int))
        end = tuple(proj1[0, 0].astype(int))
        return start, end
    except Exception:
        return None, None


def estimate_normal_multiscale(circle, depth_image, intrinsics_o3d, depth_scale):
    """
    多尺度环形采样 + RANSAC平面拟合，更鲁棒地估计深孔法向
    """
    if circle is None:
        return None, None

    cx, cy, r = circle
    fx = intrinsics_o3d.intrinsic_matrix[0, 0]
    fy = intrinsics_o3d.intrinsic_matrix[1, 1]
    cx_intr = intrinsics_o3d.intrinsic_matrix[0, 2]
    cy_intr = intrinsics_o3d.intrinsic_matrix[1, 2]

    h, w = depth_image.shape

    # 深度图预处理：双边滤波降噪
    depth_filtered = cv2.bilateralFilter(
        depth_image.astype(np.float32),
        d=5,
        sigmaColor=10,
        sigmaSpace=10
    ).astype(depth_image.dtype)

    # 多尺度环形采样
    scales = [
        (0.4, 0.8),  # 内环
        (0.8, 1.3),  # 中环
        (1.3, 1.8),  # 外环
    ]

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

        # 局部离群点过滤 (使用MAD - 中位数绝对偏差)
        if len(pts) > 50:
            median_z = np.median(pts[:, 2])
            mad = np.median(np.abs(pts[:, 2] - median_z))
            if mad > 1e-6:  # 避免除零
                threshold = 3 * mad * 1.4826  # MAD to std conversion
                pts = pts[np.abs(pts[:, 2] - median_z) < threshold]

        scale_counts.append(len(pts))
        all_pts.append(pts)

    if len(all_pts) == 0:
        return None, None

    # 合并所有尺度的点
    all_pts = np.vstack(all_pts)

    if len(all_pts) < 50:
        return None, None

    # RANSAC平面拟合(更鲁棒)
    best_normal = None
    best_inliers = 0
    ransac_iterations = 100
    inlier_threshold = 0.002  # 2mm

    for _ in range(ransac_iterations):
        # 随机采样3点
        idx = np.random.choice(len(all_pts), 3, replace=False)
        p1, p2, p3 = all_pts[idx]

        # 计算平面法向
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # 计算inliers
        d = np.abs((all_pts - p1) @ normal)
        inliers = np.sum(d < inlier_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal

    if best_normal is None or best_inliers < 50:
        # Fallback: SVD
        pts_centered = all_pts - all_pts.mean(axis=0)
        cov = pts_centered.T @ pts_centered / len(pts_centered)
        _, _, vh = np.linalg.svd(cov)
        best_normal = vh[-1, :]
        best_inliers = len(all_pts)  # 假设全部为inliers

    # 定向：指向相机正Z方向
    if best_normal[2] < 0:
        best_normal = -best_normal

    inlier_ratio = best_inliers / len(all_pts) if len(all_pts) > 0 else 0

    stats = {
        "total_points": len(all_pts),
        "inliers": best_inliers,
        "inlier_ratio": inlier_ratio,
        "scale_counts": scale_counts,
        "median_depth": float(np.median(all_pts[:, 2])),
        "depth_std": float(np.std(all_pts[:, 2]))
    }

    return best_normal, stats


def circle_center_to_pcd_coords(center_3d: np.ndarray) -> np.ndarray:
    """Convert circle center (camera coords) to point-cloud coords used by extractor."""
    return np.array([center_3d[0], -center_3d[1], -center_3d[2]])


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a noisy rotation matrix back to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def run_hole_icp():
    camera = RealSenseCamera()
    detector = CircleDetector()
    extractor = CirclePointCloudExtractor(margin=20)

    # 仅复用旋转工具，不跑ICP
    icp_register = HoleICPRegister(model_path="../pose/hole_model.STL", sample_points=2000, voxel_size=0.003)

    last_transform = None
    last_rpy = None
    last_normal = None
    alpha_pose = 0.25  # 位姿平滑
    alpha_normal = 0.15  # 法向平滑 - 对深孔更激进
    debug = True
    frame_idx = 0
    K = camera.get_camera_matrix()
    dist = camera.get_distortion_coeffs()
    intrinsics_o3d = camera.get_intrinsics()
    depth_scale = camera.depth_scale

    print("=" * 70)
    print("多尺度环形采样 + RANSAC拟合孔法向（优化版）。ESC退出。")
    print("=" * 70)

    try:
        while True:
            color_frame, depth_frame = camera.get_frame()
            if color_frame is None or depth_frame is None:
                continue

            display = color_frame.copy()
            frame_idx += 1
            circle = detector.detect(color_frame)

            circle_center_3d = None
            pcd_circle = None
            pcd_size = 0
            depth_stats = None
            normal_stats = None

            if circle is not None:
                cx, cy, r = map(int, circle)
                cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), 3)
                cv2.putText(display, f"Circle ({cx},{cy}) r={r}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                circle_center_3d = extractor.get_circle_center_3d(circle, depth_frame, camera)
                pcd_circle = extractor.extract(None, circle, color_frame, depth_frame,
                                               camera.get_intrinsics(), camera)
                if pcd_circle is not None:
                    pcd_size = len(pcd_circle.points)

                # Depth ROI stats (median etc.) for the circle center neighborhood
                radius_sample = max(3, int(r * 0.3))
                y_min = max(0, cy - radius_sample)
                y_max = min(depth_frame.shape[0], cy + radius_sample)
                x_min = max(0, cx - radius_sample)
                x_max = min(depth_frame.shape[1], cx + radius_sample)
                roi = depth_frame[y_min:y_max, x_min:x_max]
                valid = roi[roi > 0]
                if len(valid) > 0:
                    depth_stats = {
                        "count": len(valid),
                        "median": float(np.median(valid) * camera.depth_scale),
                        "min": float(np.min(valid) * camera.depth_scale),
                        "max": float(np.max(valid) * camera.depth_scale),
                        "ratio": len(valid) / roi.size,
                        "roi": roi,
                    }
            else:
                cv2.putText(display, "Circle not found", (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            rod_text = "Hole XYZ (m): ---, ---, ---"
            rpy_text = "Hole RPY (deg): ---, ---, ---"
            normal_text = "Normal: ---, ---, ---"
            quality_text = "Quality: ---"

            if circle_center_3d is not None:
                # 使用多尺度RANSAC方法估计法向
                normal, normal_stats = estimate_normal_multiscale(
                    circle, depth_frame, intrinsics_o3d, depth_scale
                )

                if normal is not None:
                    # 与上一帧同向
                    if last_normal is not None and np.dot(normal, last_normal) < 0:
                        normal = -normal

                    # 时间平滑
                    if last_normal is not None:
                        normal = last_normal * (1 - alpha_normal) + normal * alpha_normal
                        normal = normal / (np.linalg.norm(normal) + 1e-8)
                    last_normal = normal

                    R_align = icp_register.align_model_z_to_normal(normal)
                    T = np.eye(4)
                    T[:3, :3] = R_align
                    T[:3, 3] = circle_center_3d

                    # 位姿平滑
                    if last_transform is not None:
                        T[:3, 3] = last_transform[:3, 3] * (1 - alpha_pose) + T[:3, 3] * alpha_pose
                        R_blend = last_transform[:3, :3] * (1 - alpha_pose) + T[:3, :3] * alpha_pose
                        T[:3, :3] = orthonormalize(R_blend)

                    last_transform = T
                    R = T[:3, :3]
                    t = T[:3, 3]
                    rpy = icp_register.rotation_matrix_to_euler(R)
                    last_rpy = rpy

                    rod_text = f"Hole XYZ (m): {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}"
                    rpy_text = f"Hole RPY (deg): {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}"
                    normal_text = f"Normal: {normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}"

                    # 质量指示
                    if normal_stats:
                        inlier_ratio = normal_stats['inlier_ratio']
                        if inlier_ratio > 0.8:
                            quality = "Excellent"
                            quality_color = (0, 255, 0)
                        elif inlier_ratio > 0.6:
                            quality = "Good"
                            quality_color = (0, 255, 255)
                        else:
                            quality = "Poor"
                            quality_color = (0, 165, 255)
                        quality_text = f"Quality: {quality} ({inlier_ratio * 100:.1f}%)"

                    # 在2D图像上可视化法向箭头
                    start, end = project_normal_to_image(circle_center_3d, normal, K, dist, scale=0.05)
                    if start and end:
                        cv2.arrowedLine(display, start, end, (0, 255, 255), 2, tipLength=0.2)
                        cv2.putText(display, "Normal", (end[0] + 5, end[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                else:
                    cv2.putText(display, "Normal estimation failed", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif last_transform is not None and last_rpy is not None:
                t = last_transform[:3, 3]
                rod_text = f"Hole XYZ (m): {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}"
                rpy_text = f"Hole RPY (deg): {last_rpy[0]:.1f}, {last_rpy[1]:.1f}, {last_rpy[2]:.1f}"

            # 主要信息显示
            cv2.putText(display, rod_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, rpy_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, normal_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if normal_stats:
                cv2.putText(display, quality_text, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            quality_color, 2)

            # Debug信息
            if debug:
                y_offset = 118
                cv2.putText(display, f"PCD pts: {pcd_size}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

                if depth_stats:
                    dbg = f"Center depth: {depth_stats['median']:.4f}m (min/max: {depth_stats['min']:.4f}/{depth_stats['max']:.4f})"
                    cv2.putText(display, dbg, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 18
                    cv2.putText(display, f"Valid ratio: {depth_stats['ratio'] * 100:.1f}%",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 18

                    # 显示深度ROI
                    depth_roi_vis = cv2.convertScaleAbs(depth_stats["roi"], alpha=0.03)
                    depth_roi_vis = cv2.applyColorMap(depth_roi_vis, cv2.COLORMAP_JET)
                    cv2.imshow("Depth ROI (Center)", depth_roi_vis)

                if normal_stats:
                    cv2.putText(display,
                                f"Multiscale pts: {normal_stats['total_points']} (inliers: {normal_stats['inliers']})",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 18

                    scale_str = f"Scales: {normal_stats['scale_counts']}"
                    cv2.putText(display, scale_str, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 18

                    cv2.putText(display,
                                f"Ring depth: {normal_stats['median_depth']:.4f}m (std: {normal_stats['depth_std']:.4f})",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow("Hole Pose Estimation (Multiscale)", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hole_icp()