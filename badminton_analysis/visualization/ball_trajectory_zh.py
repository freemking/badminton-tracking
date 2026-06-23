import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 tk 依赖
import matplotlib.pyplot as plt
from collections import defaultdict

from ..court.mapper import CourtMapper

plt.style.use('dark_background')


class BallTrajectoryVisualizer:
    """
    羽毛球球路轨迹可视化类
    
    从 detections.jsonl 中读取 shuttlecock.image 数据，
    结合 metadata.json 的球场角点信息，将图像坐标转换为球场坐标，
    生成球路轨迹可视化图。
    """
    
    def __init__(self, detections_path, metadata_path, output_dir=None, fps=30):
        self.detections_path = detections_path
        self.metadata_path = metadata_path
        self.fps = fps
        self.court_width = 6.1
        self.court_length = 13.4
        
        if output_dir is None:
            detections_dir = os.path.dirname(os.path.abspath(detections_path))
            self.output_dir = os.path.join(detections_dir, 'ball_trajectory_visualizations')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'rally_trajectories'), exist_ok=True)
        
        # 球场线条颜色
        self.court_line_color = '#bbbbbb'
        self.ball_color = '#ffd700'  # 金色
        
        # 加载元数据获取球场角点
        self.court_mapper = None
        self._load_metadata()
        
        # 加载检测数据
        self.df = self._load_data()

    def _load_metadata(self):
        """加载 metadata.json 获取球场角点信息"""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            court_info = metadata.get('court', {})
            corners = court_info.get('corners')
            
            if corners and len(corners) == 4:
                self.court_mapper = CourtMapper(corners)
                print(f"已加载球场角点，创建坐标映射")
            else:
                print(f"警告: metadata.json 中未找到有效的球场角点")
                
        except FileNotFoundError:
            print(f"警告: 未找到 metadata.json ({self.metadata_path})，无法将球坐标转换为球场坐标")
        except Exception as e:
            print(f"加载 metadata.json 时出错: {e}")

    def _load_data(self):
        """从 detections.jsonl 加载球位置数据并转换为球场坐标"""
        try:
            rows = []
            with open(self.detections_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    
                    shuttlecock = record.get('shuttlecock', {}) or {}
                    ball_image = shuttlecock.get('image')
                    
                    frame = record.get('frame')
                    if ball_image and ball_image[0] and ball_image[1] and ball_image[0] != 0 and ball_image[1] != 0:
                        # 有效球位置
                        ball_x, ball_y = float(ball_image[0]), float(ball_image[1])
                        
                        # 如果有球场映射器，转换为球场坐标
                        court_x, court_y = None, None
                        if self.court_mapper:
                            try:
                                court_pos = self.court_mapper.image_to_court([ball_x, ball_y])
                                if court_pos is not None and len(court_pos) == 2:
                                    court_x, court_y = float(court_pos[0]), float(court_pos[1])
                            except Exception:
                                pass
                        
                        rows.append({
                            'frame': frame,
                            'image_x': ball_x,
                            'image_y': ball_y,
                            'court_x': court_x,
                            'court_y': court_y,
                            'valid': court_x is not None and 0 <= court_x <= self.court_width 
                                     and 0 <= court_y <= self.court_length,
                        })
            
            df = pd.DataFrame(rows)
            
            if df.empty:
                print("未找到任何球位置数据")
                return pd.DataFrame()
            
            print(f"\n球位置数据字段: {df.columns.tolist()}")
            print(f"共加载 {len(df)} 条球位置记录")
            
            # 检测回合（基于帧间隔）
            print("基于帧间隔检测回合...")
            frames = df['frame'].astype(int).tolist()
            gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            rally_breaks = [i+1 for i, gap in enumerate(gaps) if gap > 100]
            
            rally_segments = []
            start_idx = 0
            for break_idx in rally_breaks:
                rally_segments.append((start_idx, break_idx))
                start_idx = break_idx
            if start_idx < len(frames):
                rally_segments.append((start_idx, len(frames)))
            
            # 过滤掉太短的回合（少于 30 帧）
            rally_segments = [(s, e) for s, e in rally_segments if e - s >= 30]
            
            print(f"检测到 {len(rally_segments)} 个有效回合")
            
            # 分配回合 ID
            df['rally_id'] = 0
            for rally_id, (start, end) in enumerate(rally_segments, 1):
                mask = (df.index >= start) & (df.index < end)
                df.loc[mask, 'rally_id'] = rally_id
            
            print(f"数据转换完成，共 {len(df)} 条有效记录")
            return df
            
        except Exception as e:
            print(f"加载球位置数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _draw_court(self, ax=None):
        """绘制标准羽毛球场地"""
        if ax is not None:
            plt.sca(ax)
        
        doubles_width = self.court_width
        court_length = self.court_length
        single_offset = 0.46
        service_line = 1.98
        back_service = 0.76
        
        # 场地外框
        court_rect = plt.Rectangle((0, 0), doubles_width, court_length,
                                   fill=False, color=self.court_line_color, linewidth=3.5)
        plt.gca().add_patch(court_rect)
        
        # 单打线
        plt.plot([single_offset, single_offset], [0, court_length], self.court_line_color, linewidth=3.5)
        plt.plot([doubles_width - single_offset, doubles_width - single_offset],
                 [0, court_length], self.court_line_color, linewidth=3.5)
        
        # 网线
        plt.axhline(y=court_length/2, color=self.court_line_color, linestyle='--', linewidth=3.5)
        
        # 发球线
        plt.axhline(y=court_length/2 - service_line, color=self.court_line_color, linestyle='-', linewidth=3.5)
        plt.axhline(y=court_length/2 + service_line, color=self.court_line_color, linestyle='-', linewidth=3.5)
        
        # 后发球线
        plt.axhline(y=back_service, color=self.court_line_color, linestyle='-', linewidth=3.5)
        plt.axhline(y=court_length - back_service, color=self.court_line_color, linestyle='-', linewidth=3.5)
        
        # 中线（画到发球线）
        plt.plot([doubles_width/2, doubles_width/2], [0, court_length/2 - service_line],
                 self.court_line_color, linewidth=3.5)
        plt.plot([doubles_width/2, doubles_width/2], [court_length/2 + service_line, court_length],
                 self.court_line_color, linewidth=3.5)
        
        plt.gca().invert_yaxis()
        plt.xlim(-0.5, doubles_width + 0.5)
        plt.ylim(court_length + 0.5, -0.5)

    def _generate_rally_trajectory(self, rally_df, rally_id):
        """为单个回合生成球路轨迹图"""
        # 筛选有效的球场坐标数据
        valid_df = rally_df[rally_df['valid'] == True].copy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor='#1a1a1a')
        
        # ===== 左图：轨迹散点图 =====
        plt.sca(axes[0])
        self._draw_court()
        
        if not valid_df.empty:
            # 按时间顺序绘制球轨迹（颜色渐变）
            frames = valid_df['frame'].values
            if frames.max() > frames.min():
                norm_frames = (frames - frames.min()) / (frames.max() - frames.min())
            else:
                norm_frames = np.zeros(len(frames))
            
            colors = plt.cm.plasma(norm_frames)
            scatter = axes[0].scatter(
                valid_df['court_x'], valid_df['court_y'],
                c=norm_frames, cmap='plasma',
                alpha=0.7, s=40, edgecolors='none'
            )
            
            # 绘制轨迹连线
            if len(valid_df) > 1:
                coords = valid_df[['court_x', 'court_y']].values
                for i in range(len(coords) - 1):
                    alpha = 0.2 + 0.5 * (i / len(coords))
                    axes[0].plot(
                        [coords[i][0], coords[i+1][0]],
                        [coords[i][1], coords[i+1][1]],
                        color='white', alpha=alpha, linewidth=1.0
                    )
            
            plt.colorbar(scatter, ax=axes[0], label='时间进度', shrink=0.8)
        
        axes[0].set_title(f'回合 {int(rally_id)} 球路轨迹', color='white', fontsize=14)
        axes[0].set_xlabel('球场宽度 (米)', color='white')
        axes[0].set_ylabel('球场长度 (米)', color='white')
        axes[0].tick_params(colors='white')
        
        # ===== 右图：击球点热力图 =====
        plt.sca(axes[1])
        self._draw_court()
        
        if not valid_df.empty and len(valid_df) > 3:
            from matplotlib.colors import LinearSegmentedColormap
            heat_cmap = LinearSegmentedColormap.from_list("ball_heat", [(0,0,0,0), '#ffd700', '#ff4444'])
            
            try:
                import seaborn as sns
                sns.kdeplot(
                    x=valid_df['court_x'],
                    y=valid_df['court_y'],
                    cmap=heat_cmap,
                    fill=True,
                    alpha=0.8,
                    levels=10,
                    thresh=0.02,
                    bw_adjust=1.0,
                    ax=axes[1]
                )
            except ImportError:
                pass
        
        # 添加统计信息
        stats_text = f"回合 {int(rally_id)} 统计\n"
        stats_text += "----------------\n"
        stats_text += f"总帧数: {len(rally_df)}\n"
        stats_text += f"有效球位置帧数: {len(valid_df)}\n"
        
        if not valid_df.empty and len(valid_df) > 1:
            coords = valid_df[['court_x', 'court_y']].values
            distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            time_span = len(valid_df) / self.fps
            avg_speed = total_distance / time_span if time_span > 0 else 0
            stats_text += f"球移动总距离: {total_distance:.2f} m\n"
            stats_text += f"球平均速度: {avg_speed:.2f} m/s\n"
        
        axes[1].text(0.98, 0.5, stats_text,
                     horizontalalignment='right',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     bbox=dict(facecolor='#333333', alpha=0.8, boxstyle='round,pad=0.7', edgecolor='#666666'),
                     fontsize=12, family='monospace',
                     color='#ffffff')
        
        axes[1].set_title(f'回合 {int(rally_id)} 击球点分布', color='white', fontsize=14)
        axes[1].set_xlabel('球场宽度 (米)', color='white')
        axes[1].set_ylabel('球场长度 (米)', color='white')
        axes[1].tick_params(colors='white')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'rally_trajectories', f'rally_{int(rally_id)}_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"球路轨迹图已保存: {save_path}")
        return True

    def _generate_match_trajectory(self):
        """生成整场比赛的球路轨迹汇总图"""
        valid_df = self.df[self.df['valid'] == True].copy()
        
        if valid_df.empty:
            print("无有效球位置数据，无法生成比赛级球路图")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor='#1a1a1a')
        
        # ===== 左图：全场比赛球路散点图 =====
        plt.sca(axes[0])
        self._draw_court()
        
        # 按回合上色
        rally_ids = valid_df['rally_id'].unique()
        rally_ids = [rid for rid in rally_ids if rid > 0]
        rally_ids = sorted(rally_ids)
        
        if rally_ids:
            cmap = plt.cm.tab20
            for rid in rally_ids:
                rally_data = valid_df[valid_df['rally_id'] == rid]
                color = cmap((int(rid) - 1) % 20 / 20)
                axes[0].scatter(
                    rally_data['court_x'], rally_data['court_y'],
                    alpha=0.5, s=15, color=color,
                    label=f'回合 {int(rid)}'
                )
            if len(rally_ids) <= 10:
                axes[0].legend(loc='upper right', facecolor='#333333', edgecolor='#666666', 
                             labelcolor='white', fontsize=8)
        else:
            axes[0].scatter(valid_df['court_x'], valid_df['court_y'],
                          alpha=0.5, s=15, color=self.ball_color)
        
        axes[0].set_title('全场比赛球路轨迹总览', color='white', fontsize=14)
        axes[0].set_xlabel('球场宽度 (米)', color='white')
        axes[0].set_ylabel('球场长度 (米)', color='white')
        axes[0].tick_params(colors='white')
        
        # ===== 右图：击球点热力图 =====
        plt.sca(axes[1])
        self._draw_court()
        
        if len(valid_df) > 3:
            from matplotlib.colors import LinearSegmentedColormap
            heat_cmap = LinearSegmentedColormap.from_list("match_ball_heat", [(0,0,0,0), '#ffd700', '#ff4444'])
            try:
                import seaborn as sns
                sns.kdeplot(
                    x=valid_df['court_x'],
                    y=valid_df['court_y'],
                    cmap=heat_cmap,
                    fill=True,
                    alpha=0.8,
                    levels=12,
                    thresh=0.01,
                    bw_adjust=1.2,
                    ax=axes[1]
                )
            except ImportError:
                pass
        
        # 统计信息
        stats_text = "全场比赛统计\n"
        stats_text += "================\n"
        stats_text += f"总回合数: {len(rally_ids)}\n"
        stats_text += f"总有效球位置帧数: {len(valid_df)}\n"
        
        if len(valid_df) > 1 and rally_ids:
            total_distance = 0
            for rid in rally_ids:
                rally_data = valid_df[valid_df['rally_id'] == rid]
                if len(rally_data) > 1:
                    coords = rally_data[['court_x', 'court_y']].values
                    total_distance += np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
            stats_text += f"球移动总距离: {total_distance:.2f} m\n"
            
            total_time = len(valid_df) / self.fps
            if total_time > 0:
                stats_text += f"球平均速度: {total_distance/total_time:.2f} m/s\n"
        
        axes[1].text(0.98, 0.5, stats_text,
                     horizontalalignment='right',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     bbox=dict(facecolor='#333333', alpha=0.8, boxstyle='round,pad=0.7', edgecolor='#666666'),
                     fontsize=12, family='monospace',
                     color='#ffffff')
        
        axes[1].set_title('全场比赛击球点热力图', color='white', fontsize=14)
        axes[1].set_xlabel('球场宽度 (米)', color='white')
        axes[1].set_ylabel('球场长度 (米)', color='white')
        axes[1].tick_params(colors='white')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'match_ball_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"全场比赛球路图已保存: {save_path}")

    def visualize(self):
        """执行球路可视化"""
        if self.df.empty:
            print("无球位置数据可供可视化")
            return False

        if 'court_x' not in self.df.columns or self.df['court_x'].isna().all():
            print("注意: 无法将球位置转换为球场坐标，将使用图像坐标进行可视化")
            # 如果无法转换到球场坐标，尝试仅使用 scatter plot
            return self._visualize_image_only()

        try:
            # 生成每个回合的球路轨迹图
            rally_ids = self.df['rally_id'].unique()
            rally_count = 0
            for rid in rally_ids:
                if pd.isna(rid) or rid == 0:
                    continue
                rally_df = self.df[self.df['rally_id'] == rid]
                self._generate_rally_trajectory(rally_df, rid)
                rally_count += 1

            if rally_count == 0:
                print("未检测到有效回合")
                return False

            # 生成整场比赛汇总图
            self._generate_match_trajectory()

            return True
        except Exception as e:
            print(f"球路可视化过程出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _visualize_image_only(self):
        """仅使用图像坐标绘制球轨迹（无球场坐标系时）"""
        valid = self.df['image_x'].notna()
        if not valid.any():
            print("无有效球位置数据")
            return False

        fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a1a')
        
        frames = self.df.loc[valid, 'frame'].values
        norm_frames = (frames - frames.min()) / (frames.max() - frames.min()) if frames.max() > frames.min() else np.zeros(len(frames))
        
        scatter = ax.scatter(
            self.df.loc[valid, 'image_x'],
            self.df.loc[valid, 'image_y'],
            c=norm_frames, cmap='plasma',
            alpha=0.6, s=20, edgecolors='none'
        )
        plt.colorbar(scatter, ax=ax, label='时间进度')
        
        ax.set_title('球路轨迹（图像坐标）', color='white', fontsize=14)
        ax.set_xlabel('图像 X 坐标', color='white')
        ax.set_ylabel('图像 Y 坐标', color='white')
        ax.tick_params(colors='white')
        ax.invert_yaxis()
        
        save_path = os.path.join(self.output_dir, 'ball_trajectory_image_coords.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"球路轨迹图（图像坐标）已保存: {save_path}")
        return True


def analyze_ball_trajectory(detections_path, metadata_path, output_dir=None, fps=30):
    """
    分析球路轨迹数据并生成可视化
    
    Args:
        detections_path: detections.jsonl 文件路径
        metadata_path: metadata.json 文件路径
        output_dir: 输出目录
        fps: 视频帧率
        
    Returns:
        bool: 是否处理成功
    """
    print(f"\n分析球路轨迹数据: {detections_path}")
    
    try:
        visualizer = BallTrajectoryVisualizer(
            detections_path, metadata_path, output_dir, fps=fps
        )
        success = visualizer.visualize()
        
        if success:
            print(f"球路轨迹分析完成，可视化结果已保存至: {visualizer.output_dir}")
        else:
            print("球路轨迹分析失败（无有效数据）")
        
        return success
    except Exception as e:
        print(f"球路轨迹分析异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    from tkinter import Tk, filedialog
    
    print("请选择 detections.jsonl 文件...")
    
    try:
        root = Tk()
        root.withdraw()
        
        default_dir = "results"
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        file_path = filedialog.askopenfilename(
            title="选择检测数据文件",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            initialdir=default_dir
        )
        
        if not file_path:
            print("未选择文件，退出")
            sys.exit(0)
        
        # 自动查找同目录下的 metadata.json
        metadata_path = os.path.join(os.path.dirname(file_path), 'metadata.json')
        if not os.path.exists(metadata_path):
            print("未找到 metadata.json，请手动选择...")
            metadata_path = filedialog.askopenfilename(
                title="选择 metadata.json 文件",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=os.path.dirname(file_path)
            )
        
        success = analyze_ball_trajectory(file_path, metadata_path)
        
        if success:
            print("\n球路轨迹可视化测试完成")
        else:
            print("\n球路轨迹可视化测试失败")
            
    except Exception as e:
        print(f"\n测试错误: {e}")
    finally:
        try:
            root.destroy()
        except:
            pass
