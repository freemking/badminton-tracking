import argparse
import os

from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies



def main():
    parser = argparse.ArgumentParser(description='羽毛球比赛视频分析系统')
    parser.add_argument('--video-path', required=True, type=str, help='输入视频文件路径')
    parser.add_argument('--output-dir', default=None, type=str, help='输出目录，默认 results/<视频文件名>')
    parser.add_argument('--ball-model', default='weights/yolo11s-ball.pt', type=str, help='YOLO 羽毛球检测模型路径')
    parser.add_argument('--pose-family', default='rtmpose', choices=['rtmpose', 'rtmo', 'yolo-pose'], help='姿态模型族')
    parser.add_argument('--pose-mode', default='balanced', choices=['lightweight', 'balanced', 'performance'], help='RTMPose / RTMO 模型档位')
    parser.add_argument('--yolo-pose-model', default='yolo11n-pose.pt', type=str, help='YOLO pose 模型路径或模型名')
    parser.add_argument('--template-path', default=None, type=str, help='球场模板图像路径；不提供时会弹出文件选择框')
    parser.add_argument('--pose-roi', choices=['true', 'false'], default='true', help='是否显示姿态检测 ROI 框，默认 true')
    parser.add_argument('--display', choices=['true', 'false'], default='true', help='是否显示视频窗口，默认 true')
    parser.add_argument('--skeletons', choices=['true', 'false'], default='true', help='是否显示人体骨架，默认 true')
    parser.add_argument('--player-trajectories', choices=['true', 'false'], default='true', help='是否显示球员轨迹，默认 true')
    parser.add_argument('--court-trajectory', choices=['true', 'false'], default='true', help='是否显示球场轨迹，默认 true')
    parser.add_argument('--shuttlecock-trajectory', choices=['true', 'false'], default='true', help='是否显示羽毛球轨迹，默认 true')
    parser.add_argument('--player-stats', choices=['true', 'false'], default='true', help='是否显示球员统计信息，默认 true')
    parser.add_argument('--save-images', action='store_true', default=False, help='保存处理后的图像')
    parser.add_argument('--performance-stats', action='store_true', default=False, help='显示性能统计信息')
    parser.add_argument('--visualize-positions', choices=['true', 'false'], default='true', help='是否生成球员位置热力图和散点图，默认 true')
    parser.add_argument('--audio', choices=['true', 'false'], default='true', help='是否保留原视频音频，默认 true')
    parser.add_argument('--language', default='zh', choices=['zh', 'en'], help='选择界面语言 (zh/en)')
    parser.add_argument('--shuttlecock-max-jump', default=1000, type=int, help='球跟踪最大跳跃像素，默认 1000（增大可提高检测率）')
    parser.add_argument('--shuttlecock-prediction-gate', default=1200, type=int, help='球跟踪预测门控像素，默认 1200（增大可提高检测率）')
    parser.add_argument('--shuttlecock-max-missing', default=15, type=int, help='球跟踪最大连续丢失帧数，默认 15')
    parser.add_argument('--court-threshold', default=0.3, type=float, help='球场模板匹配阈值 (0.0-1.0)，默认 0.3')
    parser.add_argument('--ball-conf', default=0.10, type=float, help='YOLO 球检测置信度阈值，默认 0.10（降低可检出更多球）')
    parser.add_argument('--ball-box-area-ratio', default=0.015, type=float, help='球检测框最大面积占比，默认 0.015（增大可容纳更大的检测框）')
    parser.add_argument('--ball-aspect-ratio', default=8.0, type=float, help='球检测框最大宽高比，默认 8.0（增大可容纳运动模糊的椭圆框）')
    parser.add_argument('--ball-roi-padding', default=0.20, type=float, help='球检测 ROI 扩展比例，默认 0.20（增大可检测更靠近边缘的球）')
    parser.add_argument('--auto-court', choices=['true', 'false'], default='true', help='是否自动检测球场角点，默认 true。设为 false 使用手动标注')
    parser.add_argument('--pose-device', default='auto', choices=['auto', 'cpu', 'cuda'], help='姿态模型推理设备，默认 auto（自动检测 ONNX Runtime CUDA 支持）')
    parser.add_argument('--pose-backend', default='onnxruntime', help='姿态模型推理后端，默认 onnxruntime')
    args = parser.parse_args()

    load_runtime_dependencies()

    if args.language == 'en':
        from badminton_analysis.visualization.player_positions_en import analyze_player_positions
        from badminton_analysis.visualization.ball_trajectory_en import analyze_ball_trajectory
    else:
        from badminton_analysis.visualization.player_positions_zh import analyze_player_positions
        from badminton_analysis.visualization.ball_trajectory_zh import analyze_ball_trajectory

    system = BadmintonAnalysisSystem(
        args.video_path,
        show_display=args.display == 'true',
        show_skeletons=args.skeletons == 'true',
        show_player_trajectories=args.player_trajectories == 'true',
        show_court_trajectory=args.court_trajectory == 'true',
        show_shuttlecock_trajectory=args.shuttlecock_trajectory == 'true',
        show_player_stats=args.player_stats == 'true',
        show_performance_stats=args.performance_stats,
        save_images=args.save_images,
        language=args.language,
        output_dir=args.output_dir,
        ball_model_path=args.ball_model,
        template_path=args.template_path,
        pose_mode=args.pose_mode,
        pose_family=args.pose_family,
        yolo_pose_model=args.yolo_pose_model,
        show_pose_roi=args.pose_roi == 'true',
        shuttlecock_max_jump=args.shuttlecock_max_jump,
        shuttlecock_prediction_gate=args.shuttlecock_prediction_gate,
        shuttlecock_max_missing=args.shuttlecock_max_missing,
        court_threshold=args.court_threshold,
        ball_conf=args.ball_conf,
        max_box_area_ratio=args.ball_box_area_ratio,
        max_aspect_ratio=args.ball_aspect_ratio,
        roi_padding_ratio=args.ball_roi_padding,
        auto_court=args.auto_court == 'true',
        pose_device=args.pose_device,
        pose_backend=args.pose_backend,
    )

    system.keep_audio = args.audio == 'true'
    system.process_video()

    if args.visualize_positions == 'true':
        # 球员位置分析
        print("\n开始生成球员位置可视化...")
        try:
            success = analyze_player_positions(
                system.detections_path,
                os.path.join(system.save_dir, 'position_visualizations'),
                fps=system.fps
            )
            if not success:
                print("球员位置分析失败，继续后续分析...")
        except Exception as e:
            print(f"球员位置分析异常: {e}")
            print("继续后续分析...")
        print("球员位置可视化完成")

        # 球路轨迹分析（独立于球员位置分析，即使球员位置分析失败也会执行）
        print("\n开始生成球路轨迹可视化...")
        try:
            success = analyze_ball_trajectory(
                system.detections_path,
                system.metadata_path,
                os.path.join(system.save_dir, 'ball_trajectory_visualizations'),
                fps=system.fps
            )
            if success:
                print("球路轨迹可视化完成")
            else:
                print("球路轨迹可视化失败（无有效数据），请尝试增大 --shuttlecock-max-jump 和 --shuttlecock-prediction-gate 参数")
        except Exception as e:
            print(f"球路轨迹分析异常: {e}")

if __name__ == "__main__":
    main()
