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
paddledet_path = os.path.join(parent_path, 'PaddleDetection')
sys.path.insert(0, os.path.join(paddledet_path, 'deploy/python'))
sys.path.insert(0, os.path.join(paddledet_path, 'deploy/pptracking/python'))

from infer import Detector
from keypoint_infer import KeyPointDetector
from mot.tracker.jde_tracker import JDETracker
from mot.visualize import plot_tracking
from visualize import visualize_pose

def track_players_in_video(model_dir, keypoint_model_dir, video_file, device, threshold, output_dir):
    """
    Original version of the script before optimizations.
    Runs player tracking using PP-YOLOE and ByteTrack, with pose estimation.
    """
    # --- Configuration ---
    tracker_config = {
        'match_thres': 0.7,
        'track_buffer': 60,
        'conf_thres': 0.75,
        'low_conf_thres': 0.5,
        'min_box_area': 100,
        'vertical_ratio': 1.6
    }
    VISUAL_THRESHOLD = 0.2 # The hrnet threshold

    # --- Initialization ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temp_video_path = video_file 
    paddle.enable_static()
    
    # Initialize models with standard 'paddle' run_mode
    detector = Detector(
        model_dir, 
        device=device, 
        run_mode='paddle',
        batch_size=1, 
        threshold=threshold
    )
    
    keypoint_detector = KeyPointDetector(
        keypoint_model_dir, 
        device=device, 
        run_mode='paddle',
        batch_size=1
    )
    
    tracker = JDETracker(use_byte=True, **tracker_config)

    # --- Video I/O ---
    capture = cv2.VideoCapture(temp_video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height}, {fps} FPS, {frame_count} frames")

    lost_frame_threshold = fps * 5  # 5 seconds wait time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))}_original_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Tracking Loop ---
    frame_counter = 0
    target_track_id = None  # To store the ID of the player we are tracking
    frames_since_lost = 0
    with tqdm(total=frame_count, desc=f"Tracking players in {os.path.basename(video_file)}") as pbar:
        while True:
            ret, frame = capture.read()
            if not ret: break

            detector.preprocess([frame])
            results = detector.predict()
            detections_for_frame = np.empty((0, 6))
            if results is not None and 'boxes' in results and len(results['boxes']) > 0:
                # Filter for 'person' class (class_id == 0)
                detections_for_frame = results['boxes'][results['boxes'][:, 0] == 0]

            online_targets_dict = tracker.update(detections_for_frame)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []

            tracked_targets = online_targets_dict.get(0, [])

            if target_track_id is None:
                if tracked_targets:
                    # No player is being tracked yet, find the one with the highest confidence score
                    best_target = max(tracked_targets, key=lambda t: t.score)
                    target_track_id = best_target.track_id
                    online_tlwhs.append(best_target.tlwh)
                    online_ids.append(best_target.track_id)
                    online_scores.append(best_target.score)
                    frames_since_lost = 0
            else:
                # A player is already being tracked, find them by their ID
                found_target = False
                for target in tracked_targets:
                    if target.track_id == target_track_id:
                        online_tlwhs.append(target.tlwh)
                        online_ids.append(target.track_id)
                        online_scores.append(target.score)
                        found_target = True
                        frames_since_lost = 0
                        break
                if not found_target:
                    # The tracked player was lost.
                    frames_since_lost += 1
                    if frames_since_lost > lost_frame_threshold:
                        # Player lost for too long, reset to find a new one.
                        target_track_id = None
                        frames_since_lost = 0

            im_with_tracks = plot_tracking(
                frame, online_tlwhs, online_ids, scores=online_scores, frame_id=frame_counter
            )

            # --- Sequential (non-batch) Pose Estimation ---
            all_pose_results = {'keypoint': [[], []]} # To collect all results for the frame

            for i, tlwh in enumerate(online_tlwhs):
                track_id = online_ids[i]
                x, y, w, h = map(int, tlwh)

                if w > 0 and h > 0:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size > 0:
                        inputs = keypoint_detector.preprocess([crop])
                        raw_results = keypoint_detector.predict()
                        pose_results = keypoint_detector.postprocess(inputs, raw_results)
                        
                        # Adapt pose_results if it's a dict instead of a list of dicts
                        if isinstance(pose_results, dict):
                            pose_results = [pose_results]

                        # --- [FIX] Add robust check for the structure of pose_results ---
                        if (isinstance(pose_results, list) and pose_results and 
                            isinstance(pose_results[0], dict) and 'keypoint' in pose_results[0]):
                            result = pose_results[0]
                            keypoints = result['keypoint']
                            scores = result['score']
                            
                            # Translate keypoints to full frame coordinates
                            keypoints[0, :, 0] += x
                            keypoints[0, :, 1] += y
                            
                            all_pose_results['keypoint'][0].extend(keypoints)
                            all_pose_results['keypoint'][1].extend(scores)

            if all_pose_results['keypoint'][0]: # If any poses were found
                # Ensure they are numpy arrays
                all_pose_results['keypoint'][0] = np.array(all_pose_results['keypoint'][0])
                all_pose_results['keypoint'][1] = np.array(all_pose_results['keypoint'][1])

                im_with_tracks = visualize_pose(
                    im_with_tracks, 
                    all_pose_results, 
                    visual_thresh=VISUAL_THRESHOLD, 
                    returnimg=True
                )
            
            writer.write(im_with_tracks)
            pbar.update(1)
            frame_counter += 1

    writer.release()
    capture.release()
    print(f"\nOriginal tracking complete. Video saved to: {output_video_path}")

def main():
    default_model_dir = r'C:\LSTM-analytics\PaddleDetection\pretrained_models\ppyoloe_plus_crn_s_80e_coco'
    default_keypoint_model_dir = r'C:\LSTM-analytics\PaddleDetection\pretrained_models\hrnet_w32_256x192'

    parser = argparse.ArgumentParser(description='Track players in a video with pose estimation.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help='Path to the PP-YOLOE model directory.')
    parser.add_argument('--keypoint_model_dir', type=str, default=default_keypoint_model_dir,
                        help='Path to the keypoint (HRNet) model directory.')
    parser.add_argument('--video_file', type=str, 
                        default=r'C:\LSTM-analytics\NewTest81080.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    track_players_in_video(args.model_dir, args.keypoint_model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()
