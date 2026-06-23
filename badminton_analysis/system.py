import os
import tempfile
from tkinter import filedialog
import tkinter as tk
import time
import argparse


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtTrajectoryVisualizer, ShuttlecockTracker
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "good-badminton-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .court.detector import auto_detect_court as _auto_detect_court
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .detection.shuttlecock import ShuttlecockTracker as _ShuttlecockTracker
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    auto_detect_court = _auto_detect_court
    PlayerTracker = _PlayerTracker
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    ShuttlecockTracker = _ShuttlecockTracker
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION

class BadmintonAnalysisSystem:
    def __init__(self, video_path, show_display=True, 
                 show_skeletons=True, show_player_trajectories=True, 
                 show_court_trajectory=True, show_shuttlecock_trajectory=True,
                 show_player_stats=True, show_performance_stats=False, 
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/yolo11s-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True,
                 shuttlecock_max_jump=1000, shuttlecock_prediction_gate=1200,
                 shuttlecock_max_missing=15, court_threshold=0.3,
                 ball_conf=0.10, max_box_area_ratio=0.015,
                 max_aspect_ratio=8.0, roi_padding_ratio=0.20,
                 auto_court=True):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.ball_model_path = ball_model_path
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.show_pose_roi = show_pose_roi
        self.shuttlecock_max_jump = shuttlecock_max_jump
        self.shuttlecock_prediction_gate = shuttlecock_prediction_gate
        self.shuttlecock_max_missing = shuttlecock_max_missing
        self.auto_court = auto_court
        self.court_threshold = court_threshold
        self.ball_conf = ball_conf
        self.max_box_area_ratio = max_box_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.roi_padding_ratio = roi_padding_ratio


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_shuttlecock_trajectory = show_shuttlecock_trajectory
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        if not os.path.exists(self.ball_model_path):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO shuttlecock model and place it at "
                "weights/yolo11s-ball.pt, or pass its path with --ball-model."
            )
        
        if self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(model_path=self.yolo_pose_model)
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)
        self.yolo_ball_model = YOLO(self.ball_model_path)

        self.last_stats_update_frame = 0


        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('results', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.output_video_path = os.path.join(self.save_dir, f"detect_{self.video_name}.mp4")
        self.detection_writer = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        self.shuttlecock_tracker = ShuttlecockTracker(
            yolo_ball_model=self.yolo_ball_model,
            trajectory_length=30,
            show_trajectory=self.show_shuttlecock_trajectory,
            show_performance_stats=self.show_performance_stats,
            max_jump_pixels=self.shuttlecock_max_jump,
            prediction_gate_pixels=self.shuttlecock_prediction_gate,
            max_missing_frames=self.shuttlecock_max_missing,
            roi_padding_ratio=self.roi_padding_ratio,
            max_box_area_ratio=self.max_box_area_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            ball_conf=self.ball_conf,
        )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=self.show_performance_stats
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
    def process_video(self):
        """Process the input video."""
        self.start_time = time.time()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
        

        self.fps = fps
        

        template_path = self._get_template_path()
        template_gray, template_color = self._load_template(template_path, cap)
        

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = self._setup_video_writer(self.frame_width, self.frame_height, fps)


        corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
        self.court_corners = corners
        self.court_roi_corners = roi_corners

        self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
        self.detection_writer = JsonlDetectionWriter(self.detections_path)
        

        self.court_mapper = CourtMapper(corners)
        self.player_pose_visualizer.court_mapper = self.court_mapper
        self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=30,
                                          detection_writer=self.detection_writer, fps=fps)
        

        self.stats_visualizer = StatsVisualizer(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            language=self.language
        )
        
        frame_count = 0
        detect_frame_count = 0
        court_frame_count = 0
        error_frame_count = 0
        
        print(f"球场模板匹配阈值: {self.court_threshold}")
        print(f"开始逐帧处理，共 {total_frames} 帧...")
        
        progress_interval = max(1, total_frames // 20)  # 每 5% 报告一次进度

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            try:
                frame, detect_frame_count, is_court = self._process_frame(
                    frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count
                )
                if is_court:
                    court_frame_count += 1
            except Exception as e:
                error_frame_count += 1
                if error_frame_count <= 5:
                    print(f"警告: 第 {frame_count} 帧处理异常: {e}")
                # 继续处理下一帧，不中断整个分析
                continue
            
            if frame_count % progress_interval == 0:
                pct = frame_count / total_frames * 100
                print(f"进度: {frame_count}/{total_frames} ({pct:.0f}%) | 球场帧: {court_frame_count} | 检测帧: {detect_frame_count} | 异常帧: {error_frame_count}")

        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        print(f"\n处理完成:")
        print(f"视频总帧数: {frame_count}")
        print(f"球场视角帧数: {court_frame_count} ({court_frame_count/max(1,frame_count)*100:.1f}%)")
        print(f"实际检测帧数: {detect_frame_count}")
        print(f"处理异常帧数: {error_frame_count}")
        print(f"原始视频时长: {video_duration:.2f} 秒")
        print(f"处理耗时: {processing_time:.2f} 秒")
        print(f"处理速度比: {processing_time/video_duration:.2f}x")
        
        self._cleanup(cap)

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "shuttlecock": self.ball_model_path,
            },
            "court": {
                "template_path": template_path,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 6.1,
                    "length": 13.4,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
            },
        }
        write_json(self.metadata_path, metadata)

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray, threshold=self.court_threshold)
        
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0
            

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True

            self.rally_count += 1

            self.player_tracker.start_new_rally()
            

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False

            self.shuttlecock_tracker.clear_trajectory()


        if not is_court:
            return frame, detect_frame_count, is_court

        detect_frame_count += 1

        x1, y1 = roi_corners[0]
        x2, y2 = roi_corners[1]
        roi = frame[y1:y2, x1:x2]
        if self.show_pose_roi:
            cv2.rectangle(frame, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
            cv2.putText(frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        centroids, point_left_hands, point_right_hands = self.player_pose_visualizer.detect_players(roi, x1, y1)
        detected_ball_position = self.shuttlecock_tracker.detect_ball(frame, roi_corners=roi_corners)
        ball_position = self.shuttlecock_tracker.update_trajectory(detected_ball_position, roi_corners)
        

        self.shuttlecock_tracker.handle_visualization(frame)
        

        players = self.player_tracker.update(frame_count, centroids, ball_position, 
                                             point_left_hands, point_right_hands, detect_frame_count)
        

        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:

            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        t0 = time.time()

        self.player_pose_visualizer.draw_players(
            frame=frame, 
            player_tracker=self.player_tracker, 
            cached_movement_stats=self.cached_movement_stats,
            stats_visualizer=self.stats_visualizer if self.show_player_stats else None,
            rally_count=self.rally_count
        )
        t1 = time.time()
        if self.show_performance_stats:
            print(f"Drawing players took {t1 - t0:.2f} sec")
        

        if self.show_court_trajectory:
            t0 = time.time()
            frame = self.court_trajectory_visualizer.draw_overlay(frame, self.player_tracker.court_history)
            t1 = time.time()
            if self.show_performance_stats:
                print(f"Drawing court trajectory took {t1 - t0:.2f} sec")
        

        if frame is not None:
            if self.show_display:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            out.write(frame)

            if self.save_images:
                cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)
        return frame, detect_frame_count, is_court

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        
        return template_gray, template_color

    def _setup_video_writer(self, frame_width, frame_height, fps):

        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation. Try auto-detection first, fall back to manual."""

        corners = None
        roi_corners = None
        mid_height = None

        # 如果已有缓存的标注文件，直接加载
        cache_path = os.path.join(self.save_dir, 'court_annotations.txt')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                corners = eval(f.readline().split('=')[1])
                f.readline()
                mid_height = eval(f.readline().split('=')[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
            print("已加载缓存的球场标注")
        elif self.auto_court:
            # 尝试自动检测
            print("\n" + "="*50)
            print("  自动球场检测")
            print("="*50)
            corners, roi_corners, mid_height = auto_detect_court(template_color)

            if corners and len(corners) == 4:
                print("自动球场检测成功！")
                print(f"  角点: {corners}")
            else:
                print("自动检测失败，切换到手动标注模式...")
                corners, roi_corners, mid_height = annotate_court(template_color)
        else:
            corners, roi_corners, mid_height = annotate_court(template_color)

        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError(
                "Court annotation is incomplete: unable to detect or annotate court corners. "
                "Try using --auto-court false for manual annotation."
            )

        with open(cache_path, 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        return corners, roi_corners, mid_height

    def _cleanup(self, cap):
        """Clean up resources and merge audio when needed."""
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None

        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            time.sleep(1)

        cap.release()

        if self.show_display:
            cv2.destroyAllWindows()

        if hasattr(self, 'keep_audio') and self.keep_audio:
            vap.process_video_with_audio(
                video_path=self.video_path,
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path,
                save_dir=self.save_dir
            )
        else:
            vap.process_video_without_audio(
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path
            )

    def analyze_shuttlecock(self, roi_corners, corners):
        """Hit-point analysis is currently disabled."""
        raise RuntimeError(
            "Hit-point analysis is disabled until it is migrated to detections.jsonl."
        )

    def is_court_view(self, frame, template_gray, threshold=0.75):
        """Return whether the frame matches the court template."""
        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        # print("match score: ", result)
        return np.max(result) >= threshold

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay
