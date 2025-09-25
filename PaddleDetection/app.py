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

def run_video_detection(model_dir, video_file, device, threshold, output_dir):
    """
    Runs object detection on a video file and saves the output.
    """
    # Step 1: Check and convert video codec to H.264 if needed
    temp_video_path = video_file
    try:
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError("FFmpeg not found. Please install it and add to PATH.")
        
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of",
            "default=noprint_wrappers=1:nokey=1", video_file
        ]
        
        codec_name = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        print(f"Input video codec detected as: {codec_name}")
        
        if codec_name != "h264":
            print("Converting video to h264 format...")
            output_converted_path = f"output/converted_{os.path.basename(video_file).split('.')[0]}.mp4"
            convert_cmd = [
                "ffmpeg", "-i", video_file, "-vcodec", "libx264", 
                "-acodec", "aac", "-y", output_converted_path
            ]
            subprocess.run(convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_video_path = output_converted_path
            print("Conversion complete.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error during video conversion: {e}")
        return None

    # Step 2: Initialize the detector and video I/O
    paddle.enable_static()
    detector = Detector(
        model_dir,
        device=device,
        run_mode='paddle',
        batch_size=1,
        threshold=threshold,
        output_dir=output_dir
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    capture = cv2.VideoCapture(temp_video_path)
    if not capture.isOpened():
        print(f"Error: Could not open video file {temp_video_path} for processing.")
        return None
    
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height}, {fps} FPS, {frame_count} frames")

    output_video_path = os.path.join(output_dir, f"{os.path.basename(video_file).split('.')[0]}_det.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Step 3: Run the detection loop with a progress bar
    with tqdm(total=frame_count, desc=f"Processing video: {os.path.basename(video_file)}") as pbar:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            detector.preprocess([frame])
            results = detector.predict()

            # The fix: Check if results is valid before visualizing
            if results is not None:
                im = visualize_box_mask(
                    frame,
                    results,
                    detector.pred_config.labels,
                    threshold=threshold
                )
                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                writer.write(im)
            else:
                # If no detections, just write the original frame
                writer.write(frame)
                
            pbar.update(1)

    # Step 4: Release resources and clean up
    writer.release()
    capture.release()
    if temp_video_path != video_file and os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    print(f"\nProcessed video saved to: {output_video_path}")
    return output_video_path

def main():
    parser = argparse.ArgumentParser(description='Run detection on a video.')
    parser.add_argument('--model_dir', type=str, default='output_inference/ppyoloe_crn_l_300e_coco',
                        help='Path to the model directory.')
    parser.add_argument('--video_file', type=str, required=True,
                        help='Path to the video file.')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device to use: CPU or GPU.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for detection.')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output video.')
    
    args = parser.parse_args()
    
    run_video_detection(args.model_dir, args.video_file, args.device, args.threshold, args.output_dir)

if __name__ == '__main__':
    main()
