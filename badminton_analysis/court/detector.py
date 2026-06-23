"""
自动检测羽毛球场地四个角点。

策略：
1. 颜色分割（HSV 阈值）隔离场地地面
2. 轮廓查找 → 多边形近似 → 提取四边形
3. 备选：边缘检测 + 霍夫线检测 → 直线聚类 → 求交点

返回格式与 annotate_court() 一致：(corners, roi_corners, mid_height)
"""

import cv2
import numpy as np
from .mapper import compute_expanded_roi


# ============================================================
# 颜色分割参数 — 覆盖常见的羽毛球场地面颜色
# ============================================================

# 绿/蓝色场地（室内塑胶）
COURT_COLOR_RANGES = [
    # 绿色场地（HSV 下界, 上界）
    {"lower": (28, 30, 30), "upper": (85, 255, 255), "name": "green"},
    # 蓝色场地
    {"lower": (90, 30, 30), "upper": (140, 255, 255), "name": "blue"},
    # 浅蓝 / 青色
    {"lower": (80, 20, 20), "upper": (110, 255, 255), "name": "cyan"},
]

# 红色/橙色场地（室外 / 某些室内）
COURT_COLOR_RANGES_EXTRA = [
    {"lower": (0, 40, 40), "upper": (20, 255, 255), "name": "red"},
    {"lower": (160, 40, 40), "upper": (180, 255, 255), "name": "red-2"},
    {"lower": (10, 30, 30), "upper": (28, 255, 255), "name": "orange"},
    # 灰色/白色场地（线在浅色地面上）
    {"lower": (0, 0, 140), "upper": (180, 30, 255), "name": "white-bright"},
]

# 木质/米色场地
WOOD_COLOR_RANGE = {"lower": (10, 10, 80), "upper": (40, 80, 230), "name": "wood"}


def _extract_largest_quadrilateral(contours, min_area_ratio=0.05, epsilon_factor=0.02):
    """
    从轮廓列表中提取最大的近似四边形。
    
    Args:
        contours: cv2.findContours 返回的轮廓列表
        min_area_ratio: 最小面积占比（相对于图像）
        epsilon_factor: approxPolyDP 的 epsilon 因子
    
    Returns:
        4 个角点 [(x,y), ...] 或 None
    """
    if not contours:
        return None

    image_area = 1  # 会在调用处设为实际值
    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < best_area * 0.3 and best_quad is not None:
            continue

        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best_quad = approx

    if best_quad is not None:
        points = best_quad.reshape(4, 2)
        return _order_corners_clockwise(points)
    return None


def _order_corners_clockwise(points):
    """
    将四个点按顺时针排序：左上 → 右上 → 右下 → 左下
    """
    # 按 y 坐标分组
    sorted_by_y = points[np.argsort(points[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # 上面两个按 x 排序（左 → 右）
    top_two = top_two[np.argsort(top_two[:, 0])]
    # 下面两个按 x 排序（左 → 右）
    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]

    # 返回：左上, 右上, 右下, 左下
    return [
        tuple(top_two[0].astype(int).tolist()),
        tuple(top_two[1].astype(int).tolist()),
        tuple(bottom_two[1].astype(int).tolist()),
        tuple(bottom_two[0].astype(int).tolist()),
    ]


def _detect_by_color(image):
    """
    颜色分割方式检测场地。
    在所有颜色范围中尝试，选择面积最大的四边形结果。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    image_area = h * w

    best_corners = None
    best_area = 0

    all_ranges = COURT_COLOR_RANGES + COURT_COLOR_RANGES_EXTRA

    for cr in all_ranges:
        lower = np.array(cr["lower"], dtype=np.uint8)
        upper = np.array(cr["upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学操作：先闭后开，去除噪点
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.03:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > best_area:
                best_area = area
                best_corners = approx

    if best_corners is not None:
        points = best_corners.reshape(4, 2)
        return _order_corners_clockwise(points)

    return None


def _detect_by_lines(image):
    """
    边缘检测 + 霍夫线检测方式。
    找到最外层的水平线和垂直线，求交点得到角点。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值 / Canny 边缘检测
    edges = cv2.Canny(blurred, 30, 120)

    h, w = image.shape[:2]

    # 霍夫线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min(w, h) // 6,
                            maxLineGap=min(w, h) // 10)

    if lines is None or len(lines) < 4:
        return None

    horizontals = []  # 存储 (y, x1, x2)
    verticals = []    # 存储 (x, y1, y2)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # 区分水平线和垂直线
        if dy < dx * 0.3:
            # 水平线
            y_avg = (y1 + y2) / 2
            horizontals.append((y_avg, min(x1, x2), max(x1, x2)))
        elif dx < dy * 0.3:
            # 垂直线
            x_avg = (x1 + x2) / 2
            verticals.append((x_avg, min(y1, y2), max(y1, y2)))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    # 按位置排序
    horizontals.sort(key=lambda h: h[0])  # 按 y 排序
    verticals.sort(key=lambda v: v[0])     # 按 x 排序

    # 取最外侧的线
    top_y = horizontals[0][0]
    bottom_y = horizontals[-1][0]
    left_x = verticals[0][0]
    right_x = verticals[-1][0]

    # 确保有合理的间距
    if (bottom_y - top_y) < h * 0.1 or (right_x - left_x) < w * 0.1:
        return None

    # 四个角点：左上、右上、右下、左下
    corners = [
        (int(left_x), int(top_y)),
        (int(right_x), int(top_y)),
        (int(right_x), int(bottom_y)),
        (int(left_x), int(bottom_y)),
    ]

    return corners


def _detect_by_contour_approx(image):
    """
    通用轮廓近似：找到最大的四边形轮廓。
    不依赖颜色，基于灰度 + 阈值分割。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 尝试多种阈值策略
    methods = [
        ("otsu", cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive_mean", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY, 21, 5)),
        ("adaptive_gauss", cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 21, 5)),
    ]

    best_corners = None
    best_area = 0

    for name, thresh in methods:
        # 反转如果白色区域太多（场地通常是深色背景上的亮色线条）
        white_ratio = np.sum(thresh > 0) / thresh.size
        if white_ratio > 0.7:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (image.shape[0] * image.shape[1]) * 0.02:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            # 允许 4-6 个顶点的近似（有些场地角点可能被遮挡）
            if 4 <= len(approx) <= 6:
                # 找到最小外接四边形
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                # 四边形面积与轮廓面积之比应在合理范围
                if area / max(box_area, 1) > 0.7 and box_area > best_area:
                    best_area = box_area
                    # 将 box 转换为 cv2.approxPolyDP 的格式
                    best_corners = box.astype(int)

    if best_corners is not None and len(best_corners) >= 4:
        return _order_corners_clockwise(best_corners[:4])

    return None


def auto_detect_court(image):
    """
    自动检测球场角点。
    
    Args:
        image: BGR 格式的图像 (numpy array)
    
    Returns:
        (corners, roi_corners, mid_height)  — 与 annotate_court() 相同的返回格式
        或 (None, None, None) 表示检测失败
    """
    if image is None or image.size == 0:
        return None, None, None

    h, w = image.shape[:2]
    print(f"自动检测球场角点（图像尺寸: {w}x{h}）...")

    # 策略 1：颜色分割（适用于彩色场地）
    corners = _detect_by_color(image)
    if corners is not None:
        print(f"  ✓ 颜色分割成功（{len(corners)} 个角点）")
    else:
        # 策略 2：边缘 + 霍夫线检测
        print("  颜色分割失败，尝试霍夫线检测...")
        corners = _detect_by_lines(image)
        if corners is not None:
            print(f"  ✓ 霍夫线检测成功")
        else:
            # 策略 3：通用轮廓近似
            print("  霍夫线检测失败，尝试通用轮廓近似...")
            corners = _detect_by_contour_approx(image)
            if corners is not None:
                print(f"  ✓ 轮廓近似成功")

    if corners is None:
        print("  ✗ 自动检测失败，所有策略均未找到有效场地")
        return None, None, None

    # 验证角点的几何合理性
    if len(corners) != 4:
        print(f"  ✗ 角点数量异常: {len(corners)}（期望 4）")
        return None, None, None

    # 检查四边形是否凸出且面积合理
    pts = np.array(corners, dtype=np.float32)
    area = cv2.contourArea(pts)
    image_area = h * w
    if area < image_area * 0.02:
        print(f"  ✗ 检测到的四边形面积过小: {area / image_area * 100:.1f}%（最小 2%）")
        return None, None, None
    if area > image_area * 0.95:
        print(f"  ✗ 检测到的四边形面积过大: {area / image_area * 100:.1f}%（最大 95%）")
        return None, None, None

    print(f"  ✓ 自动检测成功，面积占比: {area / image_area * 100:.1f}%")

    # 计算 ROI 和 mid_height（与手动标注相同的流程）
    template_color = image.copy()
    roi_corners = compute_expanded_roi(corners, template_color.shape)

    # 通过 CourtMapper 计算 mid_height（网线位置）
    from .mapper import CourtMapper
    mapper = CourtMapper(corners)
    _, mid_height = mapper.draw_court_overlay(template_color)

    return corners, roi_corners, mid_height
