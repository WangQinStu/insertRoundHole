# rgb、灰度、降噪、边缘检测-拼接视图
import cv2
import numpy as np

import realsense.rs_camera as rs_camera

# ============
DP = 1.2              # Hough dp
MIN_DIST = 20         # 最小圆心距离（防止重复检测）
CANNY_HIGH = 150      # Canny 高阈值
HOUGH_ACC = 15        # Hough 累计阈值（越小越敏感）
MIN_RAD = 5           # 最小半径
MAX_RAD = 40          # 最大半径
CIRCULARITY_THR = 0.8  # 圆形度阈值（0~1）
# ============

def detect_circle_pipeline(frame):
    # ---- 1. 转灰度 ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- 2. 降噪并增强 ----
    blur1 = cv2.bilateralFilter(gray, 7, 60, 60)   # 双边滤波：保留边缘
    blur = cv2.GaussianBlur(blur1, (5, 5), 1.5)      # 高斯模糊：平滑噪声

    # ---- 3. 边缘检测 ----
    edges = cv2.Canny(blur, CANNY_HIGH // 2, CANNY_HIGH)

    # ---- 4. Hough 圆检测 ----
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=CANNY_HIGH,
        param2=HOUGH_ACC,
        minRadius=MIN_RAD,
        maxRadius=MAX_RAD
    )

    output = frame.copy()
    contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    detected = []

    # ---- 5. 轮廓过滤（圆形度） ----
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < CIRCULARITY_THR:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)

        detected.append((x, y, radius))
        cv2.circle(contour_img, (x, y), radius, (0, 255, 0), 2)

    # ---- 6. 在最终图上绘制 ----
    for (x, y, r) in detected:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(output, f"r={r}", (x - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- 拼接所有可视化结果 ----
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur3 = cv2.cvtColor(blur1, cv2.COLOR_GRAY2BGR)
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    row1 = np.hstack((frame, gray3, blur3))
    row2 = np.hstack((edges3, contour_img, output))
    canvas = np.vstack((row1, row2))

    return canvas, detected


# ================== 主程序：实时检测 ==================
if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)  # 改成你的 D435i RGB 流
    cap = rs_camera.RealSenseCamera()
    while True:
        frame = cap.get_color_stream()
        canvas, circles = detect_circle_pipeline(frame)
        cv2.imshow("Circle Detection Pipeline (6 Views)", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.stop()
    cv2.destroyAllWindows()
