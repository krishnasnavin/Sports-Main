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

from keypoint_infer import KeyPointDetector
from visualize import visualize_pose

def run_pose_estimation(keypoint_model_dir, video_file, device, threshold, output_dir):
    """
    Runs bottom-up pose estimation using HigherHRNet on a video.
    """
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
    
    # Load the HigherHRNet model
    keypoint_detector = KeyPointDetector(
        keypoint_model_dir,
        device=device,
        run_mode='paddle',
        batch_size=1,
        threshold=threshold, # This threshold is for the model's internal post-processing
    )

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
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}_higherhrnet_pose_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Processing Loop ---
    try:
        with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_file)} with HigherHRNet") as pbar:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                # --- Pose Estimation on the Full Frame ---
                inputs = keypoint_detector.preprocess([frame])
                raw_result = keypoint_detector.predict()
                pose_result = keypoint_detector.postprocess(inputs, raw_result)

                # --- Visualization ---
                # Reformat the result from the bottom-up model to match the visualizer's expected input format.
                if pose_result and 'keypoint' in pose_result:
                    vis_result = {
                        'keypoint': (pose_result['keypoint'], pose_result.get('score'))
                    }
                    im_with_pose = visualize_pose(
                        frame,
                        vis_result,
                        visual_thresh=threshold, # This threshold is for drawing
                        returnimg=True
                    )
                    writer.write(im_with_pose)
                else:
                    # If no poses are detected, write the original frame.
                    writer.write(frame)
                
                pbar.update(1)

    finally:
        # --- Cleanup ---
        writer.release()
        capture.release()

    print(f"\nPose estimation complete. Video saved to: {output_video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_keypoint_model_dir = os.path.join(script_dir, 'output_inference/higherhrnet_hrnet_w32_512')

    parser = argparse.ArgumentParser(description='Run HigherHRNet (bottom-up) pose estimation on a video.')
    parser.add_argument('--keypoint_model_dir', type=str, default=default_keypoint_model_dir,
                        help='Path to the HigherHRNet model directory.')
    parser.add_argument('--video_file', type=str, 
                        default=r'E:\Prototype\Test\NewTest4-720.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for keypoint visualization.')
    
    args = parser.parse_args()
    
    run_pose_estimation(args.keypoint_model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()