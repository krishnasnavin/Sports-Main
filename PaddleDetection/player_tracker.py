import os
import sys
import argparse
import cv2
import paddle
import subprocess
import shutil
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add the necessary PaddleDetection directories to the system path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, os.path.join(parent_path, 'deploy/python'))
sys.path.insert(0, os.path.join(parent_path, 'deploy/pptracking/python'))

from infer import Detector
from mot.tracker.jde_tracker import JDETracker
from mot.visualize import plot_tracking

def track_players_in_video(model_dir, video_file, device, threshold, output_dir):
    """
    Runs player tracking using PP-YOLOE and ByteTrack.
    """
    # --- Configuration ---
    tracker_config = {
        'match_thres': 0.7,
        'track_buffer': 60,
        'conf_thres': 0.5,
        'low_conf_thres': 0.5,
        'min_box_area': 100,
        'vertical_ratio': 1.6
    }

    # --- Initialization ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Video Conversion (if necessary) ---
    temp_video_path = video_file
    conversion_output_dir = "E:/Prototype/test to h264"  # Directory for converted videos

    try:
        if shutil.which("ffmpeg") is None:
            print("FFmpeg not found, skipping video conversion check. Please ensure input video is H.264.")
        else:
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of",
                "default=noprint_wrappers=1:nokey=1", video_file
            ]
            codec_name = subprocess.check_output(
                probe_cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
            
            if codec_name != "h264":
                print(f"Input video is {codec_name}, not H.264. Converting...")
                
                if not os.path.exists(conversion_output_dir):
                    os.makedirs(conversion_output_dir)

                base_name = os.path.basename(video_file)
                name_without_ext = os.path.splitext(base_name)[0]
                output_converted_path = os.path.join(conversion_output_dir, f"converted_{name_without_ext}.mp4")
                
                convert_cmd = [
                    "ffmpeg", "-i", video_file, "-vcodec", "libx264",
                    "-acodec", "aac", "-y", output_converted_path
                ]
                subprocess.run(
                    convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                temp_video_path = output_converted_path
                print(f"Conversion complete. Converted video saved to {output_converted_path}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during video conversion: {e}. Continuing with original video.")

    paddle.enable_static()
    detector = Detector(
        model_dir,
        device=device,
        run_mode='paddle',
        batch_size=1,
        threshold=threshold,
        output_dir=output_dir
    )
    # Initialize ByteTracker
    tracker = JDETracker(use_byte=True, **tracker_config)

    # --- Video I/O ---
    capture = cv2.VideoCapture(temp_video_path)
    if not capture.isOpened():
        print(f"Error: Could not open video file {temp_video_path}.")
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height}, {fps} FPS, {frame_count} frames")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}_player_tracking_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Tracking Loop ---
    frame_counter = 0
    top_n_players = 2
    tracked_player_ids = {}  # Maps internal track_id to our desired display id (1, 2, ...)
    next_player_id = 1

    try:
        with tqdm(total=frame_count, desc=f"Tracking players in {os.path.basename(video_file)}") as pbar:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                # Always run detection on every frame
                detector.preprocess([frame])
                results = detector.predict()

                detections_for_frame = np.empty((0, 6))
                if results is not None and 'boxes' in results and len(results['boxes']) > 0:
                    all_boxes = results['boxes']
                    person_class_id = 0
                    player_boxes = all_boxes[all_boxes[:, 0] == person_class_id]
                    detections_for_frame = player_boxes

                # Update tracker with detections for the current frame
                online_targets_dict = tracker.update(detections_for_frame)

                # --- Target Acquisition and Filtering ---
                filtered_tlwhs = []
                filtered_ids = []
                filtered_scores = []

                all_current_targets = []
                for cls_id in online_targets_dict:
                    all_current_targets.extend(online_targets_dict[cls_id])

                # On the first frame with detections, acquire the main players
                if not tracked_player_ids and all_current_targets:
                    all_current_targets.sort(key=lambda t: t.score, reverse=True)
                    for target in all_current_targets[:top_n_players]:
                        if next_player_id <= top_n_players:
                            tracked_player_ids[target.track_id] = next_player_id
                            next_player_id += 1
                    print(f"Initial players acquired. Mapping: {tracked_player_ids}")

                # In every frame, filter to only show the acquired players
                for target in all_current_targets:
                    internal_id = target.track_id
                    if internal_id in tracked_player_ids:
                        display_id = tracked_player_ids[internal_id]
                        tlwh = target.tlwh
                        if tlwh[2] * tlwh[3] > 0:
                            filtered_tlwhs.append(tlwh)
                            filtered_ids.append(display_id)
                            filtered_scores.append(target.score)

                # --- Visualization ---
                im_with_tracks = plot_tracking(
                    frame,
                    filtered_tlwhs,
                    filtered_ids,
                    scores=filtered_scores,
                    frame_id=frame_counter
                )

                writer.write(im_with_tracks)
                pbar.update(1)
                frame_counter += 1

    finally:
        # --- Cleanup ---
        writer.release()
        capture.release()

    print(f"\nPlayer tracking complete. Video saved to: {output_video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, 'output_inference/ppyoloe_crn_m_300e_coco')

    parser = argparse.ArgumentParser(description='Track players in a video using PP-YOLOE and ByteTrack.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help='Path to the PP-YOLOE model directory.')
    parser.add_argument('--video_file', type=str, 
                        default=r'E:\Prototype\Test\test_video3.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    
    track_players_in_video(args.model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()
