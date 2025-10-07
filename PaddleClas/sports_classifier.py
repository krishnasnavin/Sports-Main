import os
import sys
import cv2
import random
import numpy as np
import paddle.inference as paddle_infer


def create_predictor(model_dir):
    """Create a PaddlePaddle predictor from an inference model directory."""
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")
    if not os.path.exists(model_file) or not os.path.exists(params_file):
        raise FileNotFoundError(f"Model files not found in {model_dir}")

    config = paddle_infer.Config(model_file, params_file)
    config.disable_gpu()  # Use CPU
    config.enable_mkldnn()
    predictor = paddle_infer.create_predictor(config)
    return predictor


def preprocess_batch(img_list):
    """Replicate the preprocessing steps for a batch of images."""
    preprocessed_imgs = []
    for img in img_list:
        # 1. Resize
        resize_short = 256
        h, w = img.shape[:2]
        if (w <= h and w == resize_short) or (h <= w and h == resize_short):
            pass
        else:
            if w < h:
                out_w = resize_short
                out_h = int(h * resize_short / w)
            else:
                out_h = resize_short
                out_w = int(w * resize_short / h)
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        # 2. Crop
        crop_size = 224
        h, w = img.shape[:2]
        h_start = (h - crop_size) // 2
        w_start = (w - crop_size) // 2
        img = img[h_start:h_start + crop_size, w_start:w_start + crop_size, :]

        # 3. Normalize
        img = img.astype("float32") / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")
        img = (img - mean) / std

        # 4. ToCHW
        img = img.transpose((2, 0, 1))
        preprocessed_imgs.append(img)
    
    return np.array(preprocessed_imgs)


def postprocess(output, topk=1, class_id_map_file="./ppcls/utils/imagenet1k_label_list.txt"):
    """Replicate the postprocessing steps to get the top-k results."""
    with open(class_id_map_file, "r") as f:
        label_list = [line.strip() for line in f.readlines()]
    
    # Apply softmax
    x_exp = np.exp(output)
    x_sum = np.sum(x_exp)
    probabilities = x_exp / x_sum

    top_k_indices = probabilities.argsort()[::-1][:topk]
    top_k_scores = probabilities[top_k_indices]
    top_k_labels = [label_list[i] for i in top_k_indices]

    return {
        "class_ids": top_k_indices.tolist(),
        "scores": top_k_scores.tolist(),
        "label_names": top_k_labels
    }

def main():
    """
    Main function to run the classification.
    """
    # --- Configuration ---
    PADDLECLAS_ROOT = os.path.dirname(__file__)
    VIDEO_PATH = os.path.join(PADDLECLAS_ROOT, "..", "Test", "test_video - 8sec - 1080p.mp4")
    INFERENCE_MODEL_DIR = os.path.join(PADDLECLAS_ROOT, "inference_finetune")
    CLASS_ID_MAP_FILE = os.path.join(PADDLECLAS_ROOT, "Sports_Clas_data", "sports_label_list.txt")
    CONFIDENCE_THRESHOLD = 0.1 # Minimum confidence to consider a detection valid
    NUM_FRAMES_TO_SAMPLE = 5

    # --- 1. Initialize the Classifier ---
    print(f"Initializing classifier...")
    try:
        predictor = create_predictor(INFERENCE_MODEL_DIR)
    except Exception as e:
        print(f"Error during predictor creation: {e}")
        return
        
    # --- 2. Extract Random Frames ---
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < NUM_FRAMES_TO_SAMPLE:
        print(f"Warning: Video has fewer than {NUM_FRAMES_TO_SAMPLE} frames. Processing all available frames.")
        frame_indices_to_process = list(range(frame_count))
    else:
        # Select 50 unique random frames
        frame_indices_to_process = sorted(random.sample(range(frame_count), NUM_FRAMES_TO_SAMPLE))

    print(f"Selected random frames: {frame_indices_to_process}")

    frames_to_process = []
    for frame_index in frame_indices_to_process:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames_to_process.append(frame)
        else:
            print(f"Warning: Could not read frame at index {frame_index}.")
    cap.release()

    if not frames_to_process:
        print("Error: Could not read any frames from the video.")
        return

    print(f"Successfully extracted {len(frames_to_process)} frames from \"{os.path.basename(VIDEO_PATH)}\".")
    # OpenCV reads in BGR, convert to RGB for processing
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_to_process]

    # --- 3. Preprocess and Run Inference ---
    print("\n--- Classifying Frames ---")
    try:
        input_data = preprocess_batch(frames_rgb)

        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.copy_from_cpu(input_data)

        predictor.run()

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu() # Shape: (batch_size, num_classes)

        # --- 4. Analyze Results ---
        print("\n--- Classification Results ---")
        detected_sports = []
        for i, frame_output in enumerate(output_data):
            frame_idx = frame_indices_to_process[i]
            result = postprocess(frame_output, topk=1, class_id_map_file=CLASS_ID_MAP_FILE)
            
            label = result['label_names'][0]
            confidence = result['scores'][0]

            if confidence > CONFIDENCE_THRESHOLD:
                print(f"Frame {frame_idx}: Detected '{label}' with {confidence:.2f} confidence")
                detected_sports.append(label)
            else:
                print(f"Frame {frame_idx}: No sport detected with confidence > {CONFIDENCE_THRESHOLD}")

        # --- 5. Overall Result ---
        print("\n--- Overall Result ---")
        if detected_sports:
            sport_counts = {}
            for sport in detected_sports:
                sport_counts[sport] = sport_counts.get(sport, 0) + 1
            
            most_common_sport = max(sport_counts, key=sport_counts.get)
            count = sport_counts[most_common_sport]
            
            print(f"The most detected sport is '{most_common_sport}' (detected in {count} out of {len(detected_sports)} confident frames).")
        else:
            print("No sport was detected with enough confidence in the sampled frames.")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
