from collections import deque
import time

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None


class ShuttlecockTracker:
    """Detect, filter, track, and draw shuttlecock positions."""

    def __init__(
        self,
        yolo_ball_model,
        trajectory_length=30,
        show_trajectory=True,
        show_performance_stats=False,
        max_jump_pixels=1000,
        prediction_gate_pixels=1200,
        max_missing_frames=15,
        roi_padding_ratio=0.20,
        max_box_area_ratio=0.015,
        max_aspect_ratio=8.0,
        ball_conf=0.10,
        fps=30,
        court_mapper=None,
    ):
        self.yolo_ball_model = yolo_ball_model
        self.trajectory_length = trajectory_length
        self.show_trajectory = show_trajectory
        self.show_performance_stats = show_performance_stats
        self.max_jump_pixels = max_jump_pixels
        self.prediction_gate_pixels = prediction_gate_pixels
        self.max_missing_frames = max_missing_frames
        self.roi_padding_ratio = roi_padding_ratio
        self.max_box_area_ratio = max_box_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.ball_conf = ball_conf
        self.fps = fps
        self.court_mapper = court_mapper

        self.shuttlecock_trajectory = deque(maxlen=trajectory_length)
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0
        self.ball_speed_kmh = 0.0

        # 速度计算专用：记录 (frame_number, court_x_m, court_y_m)
        self._speed_points = deque(maxlen=10)
        self._ema_speed = 0.0
        self._ema_alpha = 0.4  # EMA 平滑系数

        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            self.ultra_device = 0
        else:
            self.ultra_device = "cpu"

    def detect_ball(self, frame, conf=None, roi_corners=None):
        if conf is None:
            conf = self.ball_conf
        t0 = time.time()
        try:
            ball_results = self.yolo_ball_model(frame, conf=conf, device=self.ultra_device, verbose=False)[0]
        except TypeError:
            ball_results = self.yolo_ball_model(frame, conf=conf, verbose=False)[0]

        if self.show_performance_stats:
            print(f"YOLO shuttlecock inference took {time.time() - t0:.2f} sec")

        candidates = self._extract_candidates(ball_results, frame.shape, roi_corners)
        selected = self._select_candidate(candidates)
        self.last_candidate = selected
        self.last_detection = {
            "visible": selected is not None,
            "accepted": False,
            "image": list(selected["point"]) if selected else None,
            "confidence": selected["confidence"] if selected else None,
            "candidate_count": len(candidates),
        }
        return list(selected["point"]) if selected else [0, 0]

    def update_trajectory(self, ball_position, roi_corners=None, frame_number=None):
        if ball_position == [0, 0] or ball_position is None:
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        point = tuple(ball_position)
        if not self._point_in_roi(point, roi_corners):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        if self._is_outlier(point):
            self._record_missing_detection()
            self._mark_detection_rejected()
            return [0, 0]

        self._append_valid_point(point, frame_number)
        self.last_detection["accepted"] = True
        self.last_detection["image"] = list(point)
        return list(point)

    def _extract_candidates(self, ball_results, frame_shape, roi_corners):
        boxes = ball_results.boxes
        if boxes is None or boxes.xywh.shape[0] < 1:
            return []

        xywh = boxes.xywh.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xywh))
        frame_area = max(1, frame_shape[0] * frame_shape[1])

        candidates = []
        for box, confidence in zip(xywh, confidences):
            center_x, center_y, width, height = [float(value) for value in box]
            if width <= 0 or height <= 0:
                continue

            point = (int(center_x), int(center_y))
            area_ratio = (width * height) / frame_area
            aspect_ratio = max(width / height, height / width)
            if area_ratio > self.max_box_area_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if not self._point_in_roi(point, roi_corners):
                continue

            candidates.append(
                {
                    "point": point,
                    "confidence": float(confidence),
                    "area_ratio": float(area_ratio),
                    "aspect_ratio": float(aspect_ratio),
                }
            )

        return candidates

    def _select_candidate(self, candidates):
        if not candidates:
            return None

        if not self.shuttlecock_trajectory:
            return max(candidates, key=lambda item: item["confidence"])

        predicted = self._predict_next_position()

        def score(candidate):
            distance = self._distance(candidate["point"], predicted)
            size_penalty = candidate["area_ratio"] * 4000
            return candidate["confidence"] * 1000 - distance * 1.4 - size_penalty

        return max(candidates, key=score)

    def _point_in_roi(self, point, roi_corners):
        if roi_corners is None:
            return True

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        padding = int(max(x2 - x1, y2 - y1) * self.roi_padding_ratio)
        return (x1 - padding) <= point[0] <= (x2 + padding) and (y1 - padding) <= point[1] <= (y2 + padding)

    def _is_outlier(self, point):
        if not self.shuttlecock_trajectory:
            return False

        last_point = self.shuttlecock_trajectory[-1]
        jump_distance = self._distance(point, last_point)
        strict_gate = self.missing_frames <= self.max_missing_frames
        if jump_distance > self.max_jump_pixels and strict_gate:
            return True

        predicted = self._predict_next_position()
        predicted_distance = self._distance(point, predicted)
        if predicted_distance > self.prediction_gate_pixels and strict_gate:
            return True

        return False

    def _predict_next_position(self):
        if len(self.shuttlecock_trajectory) < 2:
            return self.shuttlecock_trajectory[-1]

        prev_x, prev_y = self.shuttlecock_trajectory[-2]
        last_x, last_y = self.shuttlecock_trajectory[-1]
        return (last_x + (last_x - prev_x), last_y + (last_y - prev_y))

    def _append_valid_point(self, point, frame_number=None):
        self.shuttlecock_trajectory.append(point)
        self.last_valid_position = point
        self.missing_frames = 0

        if frame_number is None:
            return

        # 转换为球场坐标
        court_pt = None
        if self.court_mapper is not None:
            cp = self.court_mapper.image_to_court(list(point))
            if cp is not None and len(cp) >= 2 and cp[0] is not None and cp[1] is not None:
                court_pt = (float(cp[0]), float(cp[1]))

        if court_pt is not None:
            self._speed_points.append((frame_number, court_pt[0], court_pt[1]))
            self._compute_speed()
        else:
            self._compute_speed_from_pixels(point)

    def _compute_speed(self):
        """用实际帧号和球场坐标计算瞬时速度 (km/h) + EMA 平滑。"""
        pts = list(self._speed_points)
        if len(pts) < 2 or self.fps <= 0:
            return

        # 用最新两次检测计算瞬时速度
        f_prev, cx_prev, cy_prev = pts[-2]
        f_curr, cx_curr, cy_curr = pts[-1]
        frame_gap = f_curr - f_prev

        if frame_gap <= 0:
            return

        dist_m = np.hypot(cx_curr - cx_prev, cy_curr - cy_prev)
        time_sec = frame_gap / self.fps
        speed_ms = dist_m / time_sec
        raw_kmh = speed_ms * 3.6

        # 对异常值做 clamp（羽毛球最快约 500 km/h）
        raw_kmh = min(raw_kmh, 500.0)

        # EMA 平滑：frame_gap 大时降低新值权重（间隔越久越不可信）
        alpha = max(0.15, self._ema_alpha / max(1, frame_gap / 2.0))
        self._ema_speed = alpha * raw_kmh + (1 - alpha) * self._ema_speed
        self.ball_speed_kmh = round(self._ema_speed, 1)

    def _compute_speed_from_pixels(self, point):
        """降级：无 court_mapper 时的像素位移估算。"""
        traj = list(self.shuttlecock_trajectory)
        if len(traj) < 2 or self.fps <= 0:
            return
        # 用最近两个像素点
        px1, py1 = traj[-2]
        px2, py2 = point
        dist_px = np.hypot(px2 - px1, py2 - py1)
        # 粗略标定：球场 6.1m ≈ 画面中 ~600px
        scale = 6.1 / 600.0
        speed_ms = dist_px * scale / (1.0 / self.fps)
        raw_kmh = min(speed_ms * 3.6, 500.0)
        self._ema_speed = self._ema_alpha * raw_kmh + (1 - self._ema_alpha) * self._ema_speed
        self.ball_speed_kmh = round(self._ema_speed, 1)

    def get_ball_speed(self):
        """返回当前球速 (km/h)。"""
        return self.ball_speed_kmh

    def _record_missing_detection(self):
        self.missing_frames += 1
        # 丢帧时速度自然衰减（羽毛球空气阻力大，每帧减速 ~3%）
        decay = max(0.0, 1.0 - 0.03 * self.missing_frames)
        self._ema_speed *= decay
        self.ball_speed_kmh = round(self._ema_speed, 1)
        if self.missing_frames > self.max_missing_frames:
            self.last_valid_position = None

    def _mark_detection_rejected(self):
        self.last_detection["accepted"] = False
        self.last_detection["image"] = None

    def _empty_detection_state(self):
        return {
            "visible": False,
            "accepted": False,
            "image": None,
            "confidence": None,
            "candidate_count": 0,
        }

    def _distance(self, point_a, point_b):
        return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    def draw_trajectory(self, frame):
        if not self.shuttlecock_trajectory:
            return

        t0 = time.time()
        color = (87, 108, 255)
        points = list(self.shuttlecock_trajectory)

        for i, point in enumerate(points):
            radius = int(3 + (i / len(points)) * 4)
            cv2.circle(frame, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)

        latest_point = points[-1]
        cv2.circle(frame, latest_point, 6, (0, 165, 255), thickness=-1, lineType=cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing shuttlecock trajectory took {time.time() - t0:.2f} sec")

    def handle_visualization(self, frame):
        if self.show_trajectory and self.shuttlecock_trajectory:
            self.draw_trajectory(frame)

    def clear_trajectory(self):
        self.shuttlecock_trajectory.clear()
        self.last_valid_position = None
        self.last_candidate = None
        self.last_detection = self._empty_detection_state()
        self.missing_frames = 0
        self.ball_speed_kmh = 0.0
        self._ema_speed = 0.0
        self._speed_points.clear()

    def get_trajectory(self):
        return list(self.shuttlecock_trajectory)

    def get_last_detection(self):
        return dict(self.last_detection)