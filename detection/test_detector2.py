# 霍夫+轮廓双重检测，检测的圆会跳变
import cv2
import numpy as np
import realsense.rs_camera as rs_camera

# ====== 参数配置 ======
class CircleDetector:
    def __init__(self):
        # Hough参数
        self.dp = 1.2
        self.min_dist = 20
        self.canny_thresh = 100
        self.hough_acc = 80
        self.min_radius = 10
        self.max_radius = 100

        # 轮廓参数
        self.min_area = 50
        self.circularity_thresh = 0.8


# ====== 图像预处理 ======
def preprocess_image(frame):
    """图像预处理：转灰度 + 去噪"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 中值滤波去除椒盐噪声，保留边缘
    blurred = cv2.medianBlur(gray, 5)
    return blurred



# ====== Hough圆检测 ======
def detect_hough_circles(blurred, detector):
    """使用Hough变换检测圆"""
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=detector.dp,
        minDist=detector.min_dist,
        param1=detector.canny_thresh,
        param2=detector.hough_acc,
        minRadius=detector.min_radius,
        maxRadius=detector.max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        return circles  # 返回形状为 (n, 3) 的数组
    return np.array([])


# ====== 轮廓圆检测 ======
def detect_contour_circles(blurred, detector):
    """通过轮廓检测圆"""
    # 边缘检测
    edges = cv2.Canny(blurred,
                      detector.canny_thresh // 2,
                      detector.canny_thresh)

    # 查找轮廓
    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < detector.min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # 计算圆形度
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < detector.circularity_thresh:
            continue

        # 最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circles.append([int(x), int(y), int(radius)])

    return np.array(circles)


# ====== 可视化 ======
def visualize_results(frame, hough_circles, contour_circles):
    """可视化检测结果"""
    result = frame.copy()

    # 绘制Hough检测结果（蓝色）
    if len(hough_circles) > 0:
        for circle in hough_circles:
            x, y, r = circle
            cv2.circle(result, (x, y), r, (255, 0, 0), 2)
            cv2.circle(result, (x, y), 2, (255, 255, 0), 2)

    # 绘制轮廓检测结果（绿色）
    if len(contour_circles) > 0:
        for circle in contour_circles:
            x, y, r = circle
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 2)

    # 显示统计信息
    info = f"Hough: {len(hough_circles)} | Contour: {len(contour_circles)}"
    cv2.putText(result, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return result


# ====== 主流程 ======
def main():
    # 初始化
    cap = rs_camera.RealSenseCamera()
    detector = CircleDetector()

    # 创建调参窗口
    cv2.namedWindow('Parameters')
    cv2.createTrackbar('Hough Acc', 'Parameters', detector.hough_acc, 100, lambda x: None)
    cv2.createTrackbar('Circularity', 'Parameters', int(detector.circularity_thresh * 100), 100, lambda x: None)

    while True:
        # 读取帧
        frame = cap.get_color_stream()

        # 更新参数
        detector.hough_acc = cv2.getTrackbarPos('Hough Acc', 'Parameters')
        detector.circularity_thresh = cv2.getTrackbarPos('Circularity', 'Parameters') / 100.0

        # 预处理
        blurred = preprocess_image(frame)

        # 双方法检测
        hough_circles = detect_hough_circles(blurred, detector)
        contour_circles = detect_contour_circles(blurred, detector)

        # 可视化
        result = visualize_results(frame, hough_circles, contour_circles)

        # 显示
        cv2.imshow('Circle Detection', result)

        # 按键控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            print(f"Hough: {len(hough_circles)} circles")
            print(f"Contour: {len(contour_circles)} circles")

    # 清理
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()