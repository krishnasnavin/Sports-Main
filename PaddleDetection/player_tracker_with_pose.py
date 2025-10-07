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
from keypoint_infer import KeyPointDetector
from mot.tracker.jde_tracker import JDETracker
from mot.visualize import plot_tracking
from visualize import visualize_pose

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Get coordinates of intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Calculate area of intersection
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate area of union
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def track_players_in_video(model_dir, keypoint_model_dir, video_file, device, threshold, output_dir):
    """
    Runs player tracking using PP-YOLOE and ByteTrack, with pose estimation.
    Includes logic to re-acquire lost tracks.
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
    keypoint_detector = KeyPointDetector(
        keypoint_model_dir,
        device=device,
        run_mode='paddle',
        batch_size=1,
        threshold=0.5,
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
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}_player_tracking_pose_{timestamp}_picodet.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Tracking Loop ---
    frame_counter = 0
    top_n_players = 1
    tracked_player_ids = {}  # Maps internal track_id to our desired display id (1, 2, ...)
    next_player_id = 1
    last_known_tlwh = {}  # Stores the last known tlwh for each display_id
    reacquisition_iou_threshold = 0.2  # IoU threshold to re-acquire a lost track

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

                # --- Target Acquisition, Re-acquisition, and Filtering ---
                filtered_tlwhs = []
                filtered_ids = []
                filtered_scores = []

                all_current_targets = []
                for cls_id in online_targets_dict:
                    all_current_targets.extend(online_targets_dict[cls_id])

                # On the first frame with detections, acquire the main players
                if not tracked_player_ids and all_current_targets:
                    all_current_targets.sort(key=lambda t: t.tlwh[2] * t.tlwh[3], reverse=True)
                    for target in all_current_targets[:top_n_players]:
                        if next_player_id <= top_n_players:
                            tracked_player_ids[target.track_id] = next_player_id
                            last_known_tlwh[next_player_id] = target.tlwh
                            next_player_id += 1
                    print(f"Initial players acquired. Mapping: {tracked_player_ids}")

                # --- Re-acquisition logic for lost tracks ---
                current_tracked_internal_ids = set(t.track_id for t in all_current_targets)
                lost_display_ids = []
                
                # Find which of our desired players are missing in the current frame
                active_internal_ids = set(tracked_player_ids.keys())
                missing_internal_ids = active_internal_ids - current_tracked_internal_ids

                if missing_internal_ids and all_current_targets:
                    unassigned_targets = [t for t in all_current_targets if t.track_id not in tracked_player_ids]
                    
                    for internal_id in missing_internal_ids:
                        display_id = tracked_player_ids[internal_id]
                        if display_id in last_known_tlwh and unassigned_targets:
                            last_tlwh_box = last_known_tlwh[display_id]
                            
                            # Find the best candidate for re-acquisition based on IoU
                            best_candidate = None
                            max_iou = -1
                            for candidate in unassigned_targets:
                                iou = calculate_iou(last_tlwh_box, candidate.tlwh)
                                if iou > max_iou:
                                    max_iou = iou
                                    best_candidate = candidate
                            
                            # If a good enough candidate is found, re-acquire it
                            if best_candidate and max_iou > reacquisition_iou_threshold:
                                print(f"Re-acquiring Player {display_id}. Old track ID {internal_id} lost. New track ID: {best_candidate.track_id} (IoU: {max_iou:.2f})")
                                
                                # Remove the old mapping and add the new one
                                del tracked_player_ids[internal_id]
                                tracked_player_ids[best_candidate.track_id] = display_id
                                
                                # Remove the re-acquired target from the unassigned list to prevent it from being used again
                                unassigned_targets.remove(best_candidate)

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
                            # Update the last known position
                            last_known_tlwh[display_id] = tlwh
                
                # --- Pose Estimation ---
                all_keypoints = []
                all_scores = []
                if len(filtered_tlwhs) > 0:
                    for tlwh in filtered_tlwhs:
                        x, y, w, h = map(int, tlwh)
                        # Ensure the crop is valid
                        if w <= 0 or h <= 0:
                            continue
                        crop = frame[y:y+h, x:x+w]

                        if crop.size == 0:
                            continue
                        
                        # Predict keypoints for the crop
                        inputs = keypoint_detector.preprocess([crop])
                        raw_result = keypoint_detector.predict()
                        result = keypoint_detector.postprocess(inputs, raw_result)
                        
                        if result and 'keypoint' in result:
                            keypoints = result['keypoint'] # shape (1, 17, 2)
                            scores = result['score'] # shape (1, 17)
                            
                            # Translate keypoints to full frame coordinates
                            for person_kpts in keypoints:
                                for kpt in person_kpts:
                                    kpt[0] += x
                                    kpt[1] += y
                            
                            all_keypoints.extend(keypoints)
                            all_scores.extend(scores)

                # --- Visualization ---
                im_with_tracks = plot_tracking(
                    frame,
                    filtered_tlwhs,
                    filtered_ids,
                    scores=filtered_scores,
                    frame_id=frame_counter
                )

                if all_keypoints:
                    pose_results = {'keypoint': [np.array(all_keypoints), np.array(all_scores)]}
                    im_with_tracks = visualize_pose(
                        im_with_tracks,
                        pose_results,
                        visual_thresh=0.5,
                        returnimg=True
                    )

                
                writer.write(im_with_tracks)
                pbar.update(1)
                frame_counter += 1

    finally:
        # --- Cleanup ---
        writer.release()
        capture.release()

    print(f"\nPlayer tracking and pose estimation complete. Video saved to: {output_video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, 'output_inference/ppyoloe_plus_crn_m_80e_coco')
    default_keypoint_model_dir = os.path.join(script_dir, 'output_inference/tinypose_256x192')

    parser = argparse.ArgumentParser(description='Track players in a video with pose estimation.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help='Path to the PP-YOLOE model directory.')
    parser.add_argument('--keypoint_model_dir', type=str, default=default_keypoint_model_dir,
                        help='Path to the keypoint model directory.')
    parser.add_argument('--video_file', type=str, 
                        default=r'E:\Prototype\Test\NewTest6-720.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='Player detection, tracking and pose output',
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    
    track_players_in_video(args.model_dir, args.keypoint_model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()
