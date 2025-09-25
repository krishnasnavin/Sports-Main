import os
import sys
import argparse
import cv2
import paddle
import subprocess
import shutil
from tqdm import tqdm
import numpy as np

# Add the PaddleDetection deploy directory to the system path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, os.path.join(parent_path, 'deploy/python'))

from infer import Detector
from visualize import visualize_box_mask

def detect_players_in_video(model_dir, video_file, device, threshold, output_dir, frame_skip=1):
    """
    Runs player detection on a video file and saves the output.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Video Conversion (if necessary) ---
    temp_video_path = video_file
    conversion_output_dir = "E:/Prototype/test to h264"

    try:
        # (Rest of the video conversion code remains the same)
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


    # --- Detector and Video I/O Initialization ---
    paddle.enable_static()
    detector = Detector(
        model_dir,
        device=device,
        run_mode='paddle',
        batch_size=1,
        threshold=threshold,
        output_dir=output_dir
    )

    capture = cv2.VideoCapture(temp_video_path)
    if not capture.isOpened():
        print(f"Error: Could not open video file {temp_video_path}.")
        return

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height}, {fps} FPS, {frame_count} frames")

    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_file))[0]}_player_det2.avi")
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Detection Loop ---
    frame_counter = 0
    last_detection_results = None

    try:
        with tqdm(total=frame_count, desc=f"Detecting players in {os.path.basename(video_file)}") as pbar:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                # --- Better Frame Skipping Logic ---
                # Only run the detector every `frame_skip` frames.
                if frame_counter % frame_skip == 0:
                    detector.preprocess([frame])
                    last_detection_results = detector.predict()
                
                # For all frames (including skipped ones), use the last known detection results.
                results = last_detection_results
                # --- End of Frame Skipping Logic ---

                # Filter results to only include players (persons) and sports balls
                if results is not None and 'boxes' in results and len(results['boxes']) > 0:
                    all_boxes = results['boxes']
                    
                    person_class_id = 0
                    sports_ball_class_id = 32

                    player_boxes = all_boxes[
                        (all_boxes[:, 0] == person_class_id) |
                        (all_boxes[:, 0] == sports_ball_class_id)
                    ]

                    filtered_results = {'boxes': player_boxes}

                    # Visualize the filtered boxes
                    im_with_boxes = visualize_box_mask(
                        frame,
                        filtered_results,
                        detector.pred_config.labels,
                        threshold=threshold
                    )
                    bgr_output_im = np.array(im_with_boxes)
                    writer.write(bgr_output_im)
                else:
                    writer.write(frame)
                
                pbar.update(1)
                frame_counter += 1 # Increment the frame counter

    finally:
        # --- Cleanup ---
        writer.release()
        capture.release()
        if temp_video_path != video_file and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    print(f"\nPlayer detection complete. Video saved to: {output_video_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, 'output_inference/ppyoloe_plus_crn_m_80e_coco')

    parser = argparse.ArgumentParser(description='Detect players in a video.')
    parser.add_argument('--model_dir', type=str, default=default_model_dir,
                        help='Path to the model directory.')
    
    parser.add_argument('--video_file', type=str, 
                        default=r'E:\Prototype\Test\test_video.mp4',
                        help='Path to the input video file.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output video.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detection.')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Number of frames to skip between detections.')
    
    args = parser.parse_args()
    
    detect_players_in_video(args.model_dir, args.video_file, args.device, args.threshold, args.output_dir, args.frame_skip)

if __name__ == '__main__':
    main()
    
