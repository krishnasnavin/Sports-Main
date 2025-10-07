import os
os.environ['GLOG_minloglevel'] = '2' 
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
        'track_buffer': 600, # Increased to 10 seconds at 60 FPS
        'conf_thres': 0.7,
        'low_conf_thres': 0.5,
        'min_box_area': 100,
        'vertical_ratio': 1.6
    }
    VISUAL_THRESHOLD = 0.1 # The hrnet threshold
    # REID_DISTANCE_THRESHOLD = 150 # Removed fixed value

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))}_original_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Tracking Loop ---
    frame_counter = 0
    target_track_id = None  # Internal ID from JDETracker
    last_known_bbox = None # Store last known bounding box of the tracked player
    display_track_id = 1 # The ID to always display for the tracked player

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

            current_target_found = False

            if target_track_id is None:
                if tracked_targets:
                    # No player is being tracked yet, find the one with the highest confidence score
                    best_target = max(tracked_targets, key=lambda t: t.score)
                    target_track_id = best_target.track_id
                    last_known_bbox = best_target.tlwh
                    current_target_found = True
            else:
                # A player is already being tracked, try to find them by their internal ID
                for target in tracked_targets:
                    if target.track_id == target_track_id:
                        online_tlwhs.append(target.tlwh)
                        online_ids.append(display_track_id) # Always display ID 1
                        online_scores.append(target.score)
                        last_known_bbox = target.tlwh # Update last known bbox
                        current_target_found = True
                        break
                
                # If the target_track_id was not found in the current frame, attempt re-identification
                if not current_target_found and last_known_bbox is not None:
                    # Dynamically calculate REID_DISTANCE_THRESHOLD based on last known bbox size
                    dynamic_reid_threshold = 1.5 * max(last_known_bbox[2], last_known_bbox[3])

                    min_distance = float('inf')
                    reidentified_target = None

                    for target in tracked_targets:
                        # Calculate distance between current target and last known position
                        # Using center point distance for simplicity
                        current_center_x = target.tlwh[0] + target.tlwh[2] / 2
                        current_center_y = target.tlwh[1] + target.tlwh[3] / 2
                        
                        last_center_x = last_known_bbox[0] + last_known_bbox[2] / 2
                        last_center_y = last_known_bbox[1] + last_known_bbox[3] / 2

                        distance = np.sqrt((current_center_x - last_center_x)**2 + (current_center_y - last_center_y)**2)

                        if distance < dynamic_reid_threshold and distance < min_distance:
                            min_distance = distance
                            reidentified_target = target
                    
                    if reidentified_target is not None:
                        target_track_id = reidentified_target.track_id # Update internal ID
                        online_tlwhs.append(reidentified_target.tlwh)
                        online_ids.append(display_track_id) # Always display ID 1
                        online_scores.append(reidentified_target.score)
                        last_known_bbox = reidentified_target.tlwh # Update last known bbox
                        current_target_found = True # Re-identified!

            # If after all attempts, no target is found for display, ensure lists are empty
            if not current_target_found:
                online_tlwhs = []
                online_ids = []
                online_scores = []

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, r'PaddleDetection\pretrained_models\ppyoloe_plus_crn_s_80e_coco')
    default_keypoint_model_dir = os.path.join(script_dir, r'PaddleDetection\output_inference\lite_hrnet_18_256x192_coco')

    parser = argparse.ArgumentParser(description='Track players in a video with pose estimation.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help='Path to the PP-YOLOE model directory.')
    parser.add_argument('--keypoint_model_dir', type=str, default=default_keypoint_model_dir,
                        help='Path to the keypoint (HRNet) model directory.')
    parser.add_argument('--video_file', type=str, 
                        default=os.path.join(script_dir, r'sample clip\NewTest11-720.mp4'),
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'output'),
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    track_players_in_video(args.model_dir, args.keypoint_model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()



