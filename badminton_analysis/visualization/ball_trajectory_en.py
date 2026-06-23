import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from ..court.mapper import CourtMapper

plt.style.use('dark_background')


class BallTrajectoryVisualizer:
    """
    Ball trajectory visualization class.
    Reads shuttlecock.image from detections.jsonl, converts image coords
    to court coords using metadata.json court corners, and generates plots.
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
        
        self.court_line_color = '#bbbbbb'
        self.ball_color = '#ffd700'
        
        self.court_mapper = None
        self._load_metadata()
        
        self.df = self._load_data()

    def _load_metadata(self):
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            court_info = metadata.get('court', {})
            corners = court_info.get('corners')
            
            if corners and len(corners) == 4:
                self.court_mapper = CourtMapper(corners)
                print(f"Loaded court corners, coordinate mapper created")
            else:
                print(f"Warning: No valid court corners in metadata.json")
                
        except FileNotFoundError:
            print(f"Warning: metadata.json not found ({self.metadata_path})")
        except Exception as e:
            print(f"Error loading metadata.json: {e}")

    def _load_data(self):
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
                        ball_x, ball_y = float(ball_image[0]), float(ball_image[1])
                        
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
                print("No ball position data found")
                return pd.DataFrame()
            
            print(f"\nBall data fields: {df.columns.tolist()}")
            print(f"Loaded {len(df)} ball position records")
            
            print("Detecting rallies based on frame gaps...")
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
            
            # Filter out segments that are too short (fewer than 3 data points, lowered threshold for sparse detections)
            MIN_RALLY_RECORDS = 3
            rally_segments = [(s, e) for s, e in rally_segments if e - s >= MIN_RALLY_RECORDS]
            
            # Fallback: if frame-gap detection finds no valid rallies, treat all data as a single rally
            if len(rally_segments) == 0 and len(df) >= MIN_RALLY_RECORDS:
                print(f"Frame-gap detection found no valid rallies, treating all {len(df)} records as one rally")
                rally_segments = [(0, len(df))]
            
            print(f"Detected {len(rally_segments)} valid rallies")
            
            df['rally_id'] = 0
            for rally_id, (start, end) in enumerate(rally_segments, 1):
                mask = (df.index >= start) & (df.index < end)
                df.loc[mask, 'rally_id'] = rally_id
            
            print(f"Data conversion complete, {len(df)} records total")
            return df
            
        except Exception as e:
            print(f"Error loading ball position data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _draw_court(self, ax=None):
        if ax is not None:
            plt.sca(ax)
        
        doubles_width = self.court_width
        court_length = self.court_length
        single_offset = 0.46
        service_line = 1.98
        back_service = 0.76
        
        court_rect = plt.Rectangle((0, 0), doubles_width, court_length,
                                   fill=False, color=self.court_line_color, linewidth=3.5)
        plt.gca().add_patch(court_rect)
        
        plt.plot([single_offset, single_offset], [0, court_length], self.court_line_color, linewidth=3.5)
        plt.plot([doubles_width - single_offset, doubles_width - single_offset],
                 [0, court_length], self.court_line_color, linewidth=3.5)
        
        plt.axhline(y=court_length/2, color=self.court_line_color, linestyle='--', linewidth=3.5)
        
        plt.axhline(y=court_length/2 - service_line, color=self.court_line_color, linestyle='-', linewidth=3.5)
        plt.axhline(y=court_length/2 + service_line, color=self.court_line_color, linestyle='-', linewidth=3.5)
        
        plt.axhline(y=back_service, color=self.court_line_color, linestyle='-', linewidth=3.5)
        plt.axhline(y=court_length - back_service, color=self.court_line_color, linestyle='-', linewidth=3.5)
        
        plt.plot([doubles_width/2, doubles_width/2], [0, court_length/2 - service_line],
                 self.court_line_color, linewidth=3.5)
        plt.plot([doubles_width/2, doubles_width/2], [court_length/2 + service_line, court_length],
                 self.court_line_color, linewidth=3.5)
        
        plt.gca().invert_yaxis()
        plt.xlim(-0.5, doubles_width + 0.5)
        plt.ylim(court_length + 0.5, -0.5)

    def _generate_rally_trajectory(self, rally_df, rally_id):
        valid_df = rally_df[rally_df['valid'] == True].copy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor='#1a1a1a')
        
        # Left: trajectory scatter
        plt.sca(axes[0])
        self._draw_court()
        
        if not valid_df.empty:
            frames = valid_df['frame'].values
            norm_frames = (frames - frames.min()) / (frames.max() - frames.min()) if frames.max() > frames.min() else np.zeros(len(frames))
            
            scatter = axes[0].scatter(
                valid_df['court_x'], valid_df['court_y'],
                c=norm_frames, cmap='plasma',
                alpha=0.7, s=40, edgecolors='none'
            )
            
            if len(valid_df) > 1:
                coords = valid_df[['court_x', 'court_y']].values
                for i in range(len(coords) - 1):
                    alpha = 0.2 + 0.5 * (i / len(coords))
                    axes[0].plot(
                        [coords[i][0], coords[i+1][0]],
                        [coords[i][1], coords[i+1][1]],
                        color='white', alpha=alpha, linewidth=1.0
                    )
            
            plt.colorbar(scatter, ax=axes[0], label='Time Progress', shrink=0.8)
        
        axes[0].set_title(f'Rally {int(rally_id)} Ball Trajectory', color='white', fontsize=14)
        axes[0].set_xlabel('Court Width (m)', color='white')
        axes[0].set_ylabel('Court Length (m)', color='white')
        axes[0].tick_params(colors='white')
        
        # Right: heatmap
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
        
        stats_text = f"Rally {int(rally_id)} Stats\n"
        stats_text += "----------------\n"
        stats_text += f"Total Frames: {len(rally_df)}\n"
        stats_text += f"Valid Positions: {len(valid_df)}\n"
        
        if not valid_df.empty and len(valid_df) > 1:
            coords = valid_df[['court_x', 'court_y']].values
            distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
            time_span = len(valid_df) / self.fps
            avg_speed = total_distance / time_span if time_span > 0 else 0
            stats_text += f"Total Distance: {total_distance:.2f} m\n"
            stats_text += f"Avg Speed: {avg_speed:.2f} m/s\n"
        
        axes[1].text(0.98, 0.5, stats_text,
                     horizontalalignment='right',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     bbox=dict(facecolor='#333333', alpha=0.8, boxstyle='round,pad=0.7', edgecolor='#666666'),
                     fontsize=12, family='monospace',
                     color='#ffffff')
        
        axes[1].set_title(f'Rally {int(rally_id)} Hit-point Distribution', color='white', fontsize=14)
        axes[1].set_xlabel('Court Width (m)', color='white')
        axes[1].set_ylabel('Court Length (m)', color='white')
        axes[1].tick_params(colors='white')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'rally_trajectories', f'rally_{int(rally_id)}_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"Ball trajectory saved: {save_path}")
        return True

    def _generate_match_trajectory(self):
        valid_df = self.df[self.df['valid'] == True].copy()
        
        if valid_df.empty:
            print("No valid ball position data for match-level visualization")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor='#1a1a1a')
        
        # Left: match-wide scatter
        plt.sca(axes[0])
        self._draw_court()
        
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
                    label=f'Rally {int(rid)}'
                )
            if len(rally_ids) <= 10:
                axes[0].legend(loc='upper right', facecolor='#333333', edgecolor='#666666', 
                             labelcolor='white', fontsize=8)
        else:
            axes[0].scatter(valid_df['court_x'], valid_df['court_y'],
                          alpha=0.5, s=15, color=self.ball_color)
        
        axes[0].set_title('Match Ball Trajectory Overview', color='white', fontsize=14)
        axes[0].set_xlabel('Court Width (m)', color='white')
        axes[0].set_ylabel('Court Length (m)', color='white')
        axes[0].tick_params(colors='white')
        
        # Right: heatmap
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
        
        stats_text = "Match Statistics\n"
        stats_text += "================\n"
        stats_text += f"Total Rallies: {len(rally_ids)}\n"
        stats_text += f"Valid Frames: {len(valid_df)}\n"
        
        if len(valid_df) > 1 and rally_ids:
            total_distance = 0
            for rid in rally_ids:
                rally_data = valid_df[valid_df['rally_id'] == rid]
                if len(rally_data) > 1:
                    coords = rally_data[['court_x', 'court_y']].values
                    total_distance += np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
            stats_text += f"Total Distance: {total_distance:.2f} m\n"
            
            total_time = len(valid_df) / self.fps
            if total_time > 0:
                stats_text += f"Avg Speed: {total_distance/total_time:.2f} m/s\n"
        
        axes[1].text(0.98, 0.5, stats_text,
                     horizontalalignment='right',
                     verticalalignment='center',
                     transform=axes[1].transAxes,
                     bbox=dict(facecolor='#333333', alpha=0.8, boxstyle='round,pad=0.7', edgecolor='#666666'),
                     fontsize=12, family='monospace',
                     color='#ffffff')
        
        axes[1].set_title('Match Hit-point Heatmap', color='white', fontsize=14)
        axes[1].set_xlabel('Court Width (m)', color='white')
        axes[1].set_ylabel('Court Length (m)', color='white')
        axes[1].tick_params(colors='white')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'match_ball_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"Match ball trajectory saved: {save_path}")

    def visualize(self):
        if self.df.empty:
            print("No ball position data to visualize")
            return False

        if 'court_x' not in self.df.columns or self.df['court_x'].isna().all():
            print("Note: Cannot convert ball positions to court coordinates, using image coordinates")
            return self._visualize_image_only()

        try:
            rally_ids = self.df['rally_id'].unique()
            rally_count = 0
            for rid in rally_ids:
                if pd.isna(rid) or rid == 0:
                    continue
                rally_df = self.df[self.df['rally_id'] == rid]
                self._generate_rally_trajectory(rally_df, rid)
                rally_count += 1

            if rally_count == 0:
                print("No valid rallies detected")
                return False

            self._generate_match_trajectory()
            return True
        except Exception as e:
            print(f"Ball trajectory visualization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _visualize_image_only(self):
        valid = self.df['image_x'].notna()
        if not valid.any():
            print("No valid ball position data")
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
        plt.colorbar(scatter, ax=ax, label='Time Progress')
        
        ax.set_title('Ball Trajectory (Image Coordinates)', color='white', fontsize=14)
        ax.set_xlabel('Image X', color='white')
        ax.set_ylabel('Image Y', color='white')
        ax.tick_params(colors='white')
        ax.invert_yaxis()
        
        save_path = os.path.join(self.output_dir, 'ball_trajectory_image_coords.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"Ball trajectory (image coords) saved: {save_path}")
        return True


def analyze_ball_trajectory(detections_path, metadata_path, output_dir=None, fps=30):
    """
    Analyze ball trajectory data and generate visualizations.
    """
    print(f"\nAnalyzing ball trajectory data: {detections_path}")
    
    try:
        visualizer = BallTrajectoryVisualizer(
            detections_path, metadata_path, output_dir, fps=fps
        )
        success = visualizer.visualize()
        
        if success:
            print(f"Ball trajectory analysis complete, results saved to: {visualizer.output_dir}")
        else:
            print("Ball trajectory analysis failed (no valid data)")
        
        return success
    except Exception as e:
        print(f"Ball trajectory analysis exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    from tkinter import Tk, filedialog
    
    print("Please select detections.jsonl file...")
    
    try:
        root = Tk()
        root.withdraw()
        
        default_dir = "results"
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        file_path = filedialog.askopenfilename(
            title="Select detection data file",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            initialdir=default_dir
        )
        
        if not file_path:
            print("No file selected, exiting")
            sys.exit(0)
        
        metadata_path = os.path.join(os.path.dirname(file_path), 'metadata.json')
        if not os.path.exists(metadata_path):
            print("metadata.json not found, please select manually...")
            metadata_path = filedialog.askopenfilename(
                title="Select metadata.json file",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=os.path.dirname(file_path)
            )
        
        success = analyze_ball_trajectory(file_path, metadata_path)
        
        if success:
            print("\nBall trajectory visualization test completed")
        else:
            print("\nBall trajectory visualization test failed")
            
    except Exception as e:
        print(f"\nTest error: {e}")
    finally:
        try:
            root.destroy()
        except:
            pass
