import cv2
import numpy as np
import open3d as o3d
import datetime

from detection.circle_detector import CircleDetector
from pose.circle_point_extractor import CirclePointCloudExtractor
from realsense.rs_camera import RealSenseCamera
from pose.circle_point_extractor import CirclePointCloudExtractor

# 调参区域
CYL_RADIUS_RANGE = (0.0001, 0.05)
CYL_DISTANCE_THRESH = 0.003
CYL_MAX_ITERS = 2000
SAVE_DEBUG_PCD = False  # 置为 True 保存 cavity 点云
SAVE_DEBUG_PCD = False  # 置为 True 保存 cavity 点云


def ring_point_cloud(circle, depth_image, color_image, intrinsics, depth_scale, inner_ratio=0.3, outer_ratio=1.2, denoise=True):
    """
    从深度图提取环形区域点云（相机坐标系 -> Open3D点云）。
    inner_ratio: 去掉中心缺深度；outer_ratio: 控制外环半径。
    """
    if circle is None:
        return None

    cx, cy, r = circle
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx_intr = intrinsics.intrinsic_matrix[0, 2]
    cy_intr = intrinsics.intrinsic_matrix[1, 2]

    max_r = int(r * outer_ratio)
    min_r = int(r * inner_ratio)
    h, w = depth_image.shape

    ys, xs = np.ogrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask_outer = dist <= max_r
    mask_inner = dist <= min_r
    ring_mask = np.logical_and(mask_outer, ~mask_inner)

    ys_idx, xs_idx = np.where(ring_mask)
    if len(xs_idx) < 30:
        return None

    roi_depth = depth_image
    if denoise:
        # 适度中值滤波，减少毛刺/孔洞
        roi_depth = cv2.medianBlur(depth_image, 5)

    depths = roi_depth[ys_idx, xs_idx]
    valid_mask = depths > 0
    if np.count_nonzero(valid_mask) < 30:
        return None

    depths = depths[valid_mask] * depth_scale
    xs_idx = xs_idx[valid_mask]
    ys_idx = ys_idx[valid_mask]

    zs = depths
    xs_cam = (xs_idx - cx_intr) * zs / fx
    ys_cam = (ys_idx - cy_intr) * zs / fy

    pts = np.stack([xs_cam, ys_cam, zs], axis=1)

    colors = []
    if color_image is not None:
        for u, v in zip(xs_idx, ys_idx):
            b, g, r_color = color_image[v, u]
            colors.append([r_color / 255.0, g / 255.0, b / 255.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def estimate_plane_and_inliers(pcd, distance_threshold=0.015):
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=80)
        return plane_model, inliers
    except Exception:
        return None, []


def fit_cylinder_ransac(pcd, radius_range=(0.001, 0.02), max_iterations=2000, distance_threshold=0.001):
    """
    简易RANSAC拟合圆柱（基于Open3D 0.17+支持）。返回axis, center, radius。
    若当前Open3D无该API，调用时会异常，调用端会用其他方法兜底。
    """
    try:
        model, inliers = pcd.segment_cylinder(distance_threshold=distance_threshold,
                                              radius_range=radius_range,
                                              max_iterations=max_iterations)
        # Open3D cylinder model: center, direction, radius
        center = np.array(model.center)
        axis = np.array(model.direction)
        radius = model.radius
        return center, axis, radius, inliers
    except Exception:
        return None, None, None, []


def main():
    camera = RealSenseCamera()
    detector = CircleDetector()
    extractor = CirclePointCloudExtractor(margin=20)

    intrinsics = camera.get_intrinsics()
    depth_scale = camera.depth_scale
    K = camera.get_camera_matrix()
    dist = camera.get_distortion_coeffs()

    # 可视化点云窗口（显示 cavity）
    vis = o3d.visualization.Visualizer()
    vis.create_window("Cavity PointCloud", width=480, height=360)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(vis_pcd)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis.add_geometry(coord)

    print("=" * 70)
    print("圆周环点云 + 去平面 + 拟合圆柱 -> 孔轴方向 (ESC退出)")
    print("=" * 70)

    try:
        last_axis = None
        alpha_axis = 0.3
        last_center_cam = None

        while True:
            color_frame, depth_frame = camera.get_frame()
            if color_frame is None or depth_frame is None:
                continue

            display = color_frame.copy()
            circle = detector.detect(color_frame)

            axis = None
            cyl_radius = None
            pcd_size = 0
            inlier_ratio = 0.0
            cavity_pts = 0
            mode_text = "Mode: --"
            cyl_inliers = 0

            if circle is not None:
                cx, cy, r = map(int, circle)
                cv2.circle(display, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 2, (0, 0, 255), 3)

                pcd_ring = ring_point_cloud(circle, depth_frame, color_frame, intrinsics, depth_scale, inner_ratio=0.2, outer_ratio=1.5)
                if pcd_ring is not None:
                    pcd_size = len(pcd_ring.points)

                    # 去掉大平面（外表面）
                    plane_model, plane_inliers = estimate_plane_and_inliers(pcd_ring, distance_threshold=0.0015)
                    if plane_model is not None and len(plane_inliers) > 0:
                        pcd_cavity = pcd_ring.select_by_index(plane_inliers, invert=True)
                    else:
                        pcd_cavity = pcd_ring

                    cavity_pts = len(pcd_cavity.points)
                    if SAVE_DEBUG_PCD and cavity_pts > 0:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"cavity_{ts}.ply"
                        o3d.io.write_point_cloud(filename, pcd_cavity)
                        print(f"[debug] saved cavity cloud: {filename} (pts={cavity_pts})")
                    if cavity_pts > 30:
                        # 先做一次体素降采样，减少噪声
                        pcd_cavity_ds = pcd_cavity.voxel_down_sample(0.0015)
                        if len(pcd_cavity_ds.points) < 30:
                            pcd_cavity_ds = pcd_cavity

                        center, axis_vec, radius, inliers_cyl = fit_cylinder_ransac(
                            pcd_cavity_ds,
                            radius_range=CYL_RADIUS_RANGE,
                            max_iterations=CYL_MAX_ITERS,
                            distance_threshold=CYL_DISTANCE_THRESH
                        )
                        mode_text = "Mode: cylinder"
                        if axis_vec is None:
                            # Fallback: 使用平面法向
                            if plane_model is not None:
                                axis_vec = np.array(plane_model[:3])
                            else:
                                # 或用 PCA 最小特征向量
                                pts = np.asarray(pcd_cavity.points)
                                pts_c = pts - pts.mean(axis=0)
                                _, _, vh = np.linalg.svd(pts_c)
                                axis_vec = vh[-1, :]
                            radius = radius if radius is not None else 0.0
                            inliers_cyl = []
                            mode_text = "Mode: fallback"

                        if axis_vec is not None:
                            # 保证指向相机正Z，并与上一帧同向
                            if axis_vec[2] < 0:
                                axis_vec = -axis_vec
                            if last_axis is not None and np.dot(axis_vec, last_axis) < 0:
                                axis_vec = -axis_vec
                            if last_axis is not None:
                                axis_vec = last_axis * (1 - alpha_axis) + axis_vec * alpha_axis
                                axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-8)
                            last_axis = axis_vec
                            axis = axis_vec
                            cyl_radius = radius
                            if inliers_cyl:
                                cyl_inliers = len(inliers_cyl)
                                inlier_ratio = cyl_inliers / len(pcd_cavity.points)
                                print(f"[cylinder] cavity_pts={cavity_pts}, inliers={cyl_inliers}, ratio={inlier_ratio:.3f}")
                            else:
                                print(f"[fallback] cavity_pts={cavity_pts}, mode={mode_text}")

                            # 在2D上画轴向箭头
                            center_cam = extractor.get_circle_center_3d(circle, depth_frame, camera)
                            if center_cam is not None:
                                last_center_cam = center_cam
                            if center_cam is None and last_center_cam is not None:
                                center_cam = last_center_cam
                            if center_cam is not None:
                                p0 = center_cam.reshape(1, 1, 3)
                                p1 = (center_cam + axis * 0.05).reshape(1, 1, 3)
                                proj0, _ = cv2.projectPoints(p0, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
                                proj1, _ = cv2.projectPoints(p1, np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
                                start = tuple(proj0[0, 0].astype(int))
                                end = tuple(proj1[0, 0].astype(int))
                                cv2.arrowedLine(display, start, end, (0, 255, 255), 2, tipLength=0.2)
                                cv2.putText(display, "Axis", (end[0] + 5, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            axis_text = "Axis: ---, ---, ---"
            radius_text = "Radius: ---"
            inlier_text = f"Inlier ratio: {inlier_ratio*100:.1f}% (inliers: {cyl_inliers})"
            if axis is not None:
                axis_text = f"Axis: {axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}"
            if cyl_radius is not None:
                radius_text = f"Radius: {cyl_radius*1000:.1f} mm"

            cv2.putText(display, axis_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, radius_text, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, inlier_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(display, f"Ring pts: {pcd_size}", (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, f"Cavity pts: {cavity_pts}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, mode_text, (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Hole Axis via Cylinder RANSAC", display)
            # 更新点云可视化
            if cavity_pts > 0:
                vis_pcd.points = pcd_cavity.points
                if pcd_cavity.has_colors():
                    vis_pcd.colors = pcd_cavity.colors
                vis.update_geometry(vis_pcd)
            vis.poll_events()
            vis.update_renderer()

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()
