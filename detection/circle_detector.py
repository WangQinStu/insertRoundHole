# 可视化edge

import cv2
import numpy as np
from realsense.rs_camera import RealSenseCamera

class CircleDetector:
    def __init__(self):
        # --- 基本参数 ---
        self.canny_low = 50
        self.canny_high = 120
        self.min_area = 40
        self.circularity_thresh = 0.75

        # --- 稳定化参数 ---
        self.alpha = 0.3  # EMA 滤波系数
        self.prev_circle = None  # (cx, cy, r)

        # --- 连续帧中值滤波 ---
        self.buffer_size = 5
        self.cx_buf, self.cy_buf, self.r_buf = [], [], []

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        return blurred

    # -------------------------------
    #   轮廓检测 + 圆形度过滤
    # -------------------------------
    def detect_contours(self, blurred):
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < self.circularity_thresh:
                continue

            # --- 使用椭圆拟合代替最小外接圆（更稳） ---
            if len(cnt) < 5:
                continue  # fitEllipse 需要 ≥5 点

            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            r = (MA + ma) / 4  # 将椭圆近似为圆，取平均半径

            candidates.append((cx, cy, r))

        return candidates

    # -------------------------------------
    #   从若干候选中选择最接近上一帧的
    # -------------------------------------
    def select_best_circle(self, circles):
        if len(circles) == 0:
            return None

        if self.prev_circle is None:
            return circles[0]  # 第一次检测直接取第一个

        px, py, pr = self.prev_circle

        dists = [(cx - px) ** 2 + (cy - py) ** 2 for cx, cy, r in circles]

        return circles[np.argmin(dists)]

    # ----------------------------------------
    #   EMA 滤波 + 多帧中值滤波（强稳定）
    # ----------------------------------------
    def smooth_circle(self, cx, cy, r):
        # --- 多帧 buffer ---
        self.cx_buf.append(cx)
        self.cy_buf.append(cy)
        self.r_buf.append(r)

        if len(self.cx_buf) > self.buffer_size:
            self.cx_buf.pop(0)
            self.cy_buf.pop(0)
            self.r_buf.pop(0)

        # --- 中值值（抗抖动） ---
        cx_med = np.median(self.cx_buf)
        cy_med = np.median(self.cy_buf)
        r_med = np.median(self.r_buf)

        # --- 指数滑动平均（工业常用） ---
        if self.prev_circle is None:
            smoothed = (cx_med, cy_med, r_med)
        else:
            px, py, pr = self.prev_circle
            smoothed = (
                px * (1 - self.alpha) + cx_med * self.alpha,
                py * (1 - self.alpha) + cy_med * self.alpha,
                pr * (1 - self.alpha) + r_med * self.alpha
            )

        self.prev_circle = smoothed
        return smoothed

    # ----------------------------------------
    #   主检测函数（外部调用）
    # ----------------------------------------
    def detect(self, frame):
        blurred = self.preprocess(frame)
        candidates = self.detect_contours(blurred)
        best = self.select_best_circle(candidates)

        if best is None:
            return None  # 没检测到

        cx, cy, r = best
        return self.smooth_circle(cx, cy, r)


# ========================================================
#                实时示例（可直接运行）
# ========================================================
def main():

    cap = RealSenseCamera()
    detector = CircleDetector()
    while True:
        frame = cap.get_color_stream()

        result = frame.copy()

        circle = detector.detect(frame)

        if circle is not None:
            cx, cy, r = map(int, circle)
            cv2.circle(result, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 2, (0, 0, 255), 2)
            cv2.putText(result, f"{cx},{cy},r={r}",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)

        cv2.imshow("Stable Circle Detection", result)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
