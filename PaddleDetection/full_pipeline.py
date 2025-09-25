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

# --- Real Sport Classification using PaddleVideo ---
def classify_sport(video_file, video_model_config, video_model_dir):
    """
    Runs sport classification using an exported PaddleVideo inference model.
    It calls the official predict.py script as a subprocess with the correct arguments.
    """
    print("Running sport classification...")
    python_executable = os.path.join(parent_path, '../paddlev/Scripts/python.exe')
    paddlevideo_script = os.path.join(parent_path, '../PaddleVideo/tools/predict.py')
    model_file = os.path.join(video_model_dir, 'ppTSM.pdmodel')
    params_file = os.path.join(video_model_dir, 'ppTSM.pdiparams')

    command = [
        python_executable,
        paddlevideo_script,
        '--config', video_model_config,
        '--input_file', video_file,
        '--model_file', model_file,
        '--params_file', params_file,
        '--use_gpu=True'
    ]

    try:
        result_text = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        top_line = [line for line in result_text.split('\n') if "top1:" in line]
        if top_line:
            class_name = top_line[0].split('class: ')[1].split(',')[0]
            print(f"Sport classified as: {class_name}")
            return class_name
        else:
            print("Could not parse sport classification result.")
            return "Unknown"

    except subprocess.CalledProcessError as e:
        print(f"Error during sport classification: {e.output}")
        return "Error"

def run_full_pipeline(model_dir, keypoint_model_dir, video_model_config, video_model_dir, video_file, device, threshold, output_dir):
    """
    Runs a full analysis pipeline:
    1. Sport Classification
    2. Player, Ball, and Racket Tracking
    3. Pose Estimation
    """
    # --- 1. Sport Classification ---
    sport_name = classify_sport(video_file, video_model_config, video_model_dir)

    # --- Configuration ---
    CLASS_IDS = {'person': 0, 'sports ball': 32, 'tennis racket': 38}
    tracker_config = {
        'match_thres': 0.7, 'track_buffer': 60, 'conf_thres': 0.5,
        'low_conf_thres': 0.5, 'min_box_area': 100, 'vertical_ratio': 1.6
    }

    # --- Initialization ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Video Conversion ---
    temp_video_path = video_file
    try:
        if shutil.which("ffmpeg") is None:
            print("FFmpeg not found, skipping video conversion check.")
        else:
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", video_file]
            codec_name = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT, text=True).strip()
            if codec_name != "h264":
                print(f"Input video is {codec_name}, not H.264. Converting...")
                conversion_output_dir = "E:/Prototype/test to h264"
                if not os.path.exists(conversion_output_dir):
                    os.makedirs(conversion_output_dir)
                base_name = os.path.basename(video_file)
                name_without_ext = os.path.splitext(base_name)[0]
                output_converted_path = os.path.join(conversion_output_dir, f"converted_{name_without_ext}.mp4")
                convert_cmd = ["ffmpeg", "-i", video_file, "-vcodec", "libx264", "-acodec", "aac", "-y", output_converted_path]
                subprocess.run(convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                temp_video_path = output_converted_path
                print(f"Conversion complete. Converted video saved to {output_converted_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during video conversion: {e}. Continuing with original video.")

    paddle.enable_static()
    detector = Detector(model_dir, device=device, run_mode='paddle', batch_size=1, threshold=threshold, output_dir=output_dir)
    keypoint_detector = KeyPointDetector(keypoint_model_dir, device=device, run_mode='paddle', batch_size=1, threshold=0.5)
    tracker = JDETracker(use_byte=True, **tracker_config)

    # --- Video I/O ---
    capture = cv2.VideoCapture(temp_video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height}, {fps} FPS, {frame_count} frames")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}_full_pipeline_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Main Processing Loop ---
    try:
        with tqdm(total=frame_count, desc=f"Running Full Pipeline on {os.path.basename(video_file)}") as pbar:
            while True:
                ret, frame = capture.read()
                if not ret: break

                # --- 2. Player & Equipment Detection ---
                detector.preprocess([frame])
                results = detector.predict()

                detections_for_frame = np.empty((0, 6))
                if results and 'boxes' in results and len(results['boxes']) > 0:
                    all_boxes = results['boxes']
                    target_class_ids = list(CLASS_IDS.values())
                    mask = np.isin(all_boxes[:, 0], target_class_ids)
                    detections_for_frame = all_boxes[mask]

                # --- Tracking ---
                online_targets_dict = tracker.update(detections_for_frame)
                
                all_tlwhs, all_ids, all_scores, player_tlwhs = [], [], [], []
                for cls_id in online_targets_dict:
                    for t in online_targets_dict[cls_id]:
                        tlwh, tid = t.tlwh, t.track_id
                        if tlwh[2] * tlwh[3] > 0:
                            all_tlwhs.append(tlwh)
                            all_ids.append(tid)
                            all_scores.append(t.score)
                            if cls_id == CLASS_IDS['person']:
                                player_tlwhs.append(tlwh)

                # --- 3. Pose Estimation (only on players) ---
                all_keypoints, all_kpt_scores = [], []
                if player_tlwhs:
                    for tlwh in player_tlwhs:
                        x, y, w, h = map(int, tlwh)
                        if w <= 0 or h <= 0: continue
                        crop = frame[y:y+h, x:x+w]
                        
                        inputs = keypoint_detector.preprocess([crop])
                        raw_kpt_result = keypoint_detector.predict()
                        kpt_result = keypoint_detector.postprocess(inputs, raw_kpt_result)
                        
                        if kpt_result and 'keypoint' in kpt_result:
                            keypoints, scores = kpt_result['keypoint'], kpt_result['score']
                            for person_kpts in keypoints:
                                for kpt in person_kpts:
                                    kpt[0] += x
                                    kpt[1] += y
                                all_keypoints.extend(keypoints)
                                all_kpt_scores.extend(scores)

                # --- Visualization ---
                im_with_tracks = plot_tracking(frame, all_tlwhs, all_ids, scores=all_scores, frame_id=pbar.n)

                if all_keypoints:
                    pose_results = {'keypoint': [np.array(all_keypoints), np.array(all_kpt_scores)]}
                    im_with_tracks = visualize_pose(im_with_tracks, pose_results, visual_thresh=0.5, returnimg=True)
                
                cv2.putText(im_with_tracks, f"Sport: {sport_name}", (width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                writer.write(im_with_tracks)
                pbar.update(1)

    finally:
        writer.release()
        capture.release()
        print(f"\nFull pipeline complete. Video saved to: {output_video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define default paths for all models
    default_model_dir = os.path.join(script_dir, 'output_inference/ppyoloe_plus_crn_m_80e_coco')
    default_keypoint_model_dir = os.path.join(script_dir, 'output_inference/tinypose_256x192')
    default_video_model_config = os.path.join(script_dir, '../PaddleVideo/configs/recognition/pptsm/pptsm_k400_videos_uniform.yaml')
    default_video_model_dir = os.path.join(script_dir, '../PaddleVideo/inference_output')

    parser = argparse.ArgumentParser(description='Run a full sports analysis pipeline.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir, help='Path to the object detection model directory.')
    parser.add_argument('--keypoint_model_dir', type=str, default=default_keypoint_model_dir, help='Path to the keypoint model directory.')
    parser.add_argument('--video_model_config', type=str, default=default_video_model_config, help='Path to the PaddleVideo model config file.')
    parser.add_argument('--video_model_dir', type=str, default=default_video_model_dir, help='Path to the PaddleVideo inference model directory.')
    parser.add_argument('--video_file', type=str, default=r'E:\Prototype\Test\test_video3.mp4', help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU', help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection.')
    
    args = parser.parse_args()
    
    run_full_pipeline(args.model_dir, args.keypoint_model_dir, args.video_model_config, args.video_model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()
