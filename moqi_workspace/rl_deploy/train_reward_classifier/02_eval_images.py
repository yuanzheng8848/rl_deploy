import os
import cv2
import glob
import numpy as np
import jax
from absl import app, flags
from tqdm import tqdm
import sys
from pathlib import Path

# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "serl" / "serl_launcher"))
sys.path.append(str(ROOT_DIR / "serl" / "serl_robot_infra"))
# ------------------

from serl_launcher.networks.reward_classifier import load_classifier_func

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "../classifier_ckpt", "Path to the classifier checkpoint")

RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/record_data"
SUCCESS_SESSION = "session_0000_27hz_20251228_125236"

def load_images_from_folder(folder_path):
    images = []
    if not os.path.exists(folder_path):
        return images
    
    # Find all images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_paths.sort()
    return image_paths

def load_and_process_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    # Add batch and time dimensions: (1, 1, 128, 128, 3)
    img = img[None, None, ...]
    return img

def evaluate_images(classifier_func, image_paths, label_name):
    if not image_paths:
        print(f"No images found for {label_name}")
        return

    print(f"Evaluating {len(image_paths)} images for {label_name}...")
    
    probs = []
    correct_count = 0
    
    for path in tqdm(image_paths):
        try:
            img = load_and_process_image(path)
            if img is None:
                continue
            
            input_obs = {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32)
            }
            
            reward = classifier_func(input_obs)
            reward_val = reward.item()
            sigmoid_prob = 1 / (1 + np.exp(-reward_val))
            probs.append(sigmoid_prob)
            
            # Check accuracy
            if label_name == "SUCCESS":
                if sigmoid_prob >= 0.5:
                    correct_count += 1
            else: # FAILURE
                if sigmoid_prob < 0.5:
                    correct_count += 1
            
            # Optional: Print individual low confidence predictions
            # if (label_name == "SUCCESS" and sigmoid_prob < 0.5) or (label_name == "FAILURE" and sigmoid_prob >= 0.5):
            #      print(f"Misclassified {os.path.basename(path)}: Prob={sigmoid_prob:.4f}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if probs:
        avg_prob = np.mean(probs)
        accuracy = correct_count / len(probs) * 100
        print(f"\n--- {label_name} Results ---")
        print(f"Total Images: {len(probs)}")
        print(f"Average Probability: {avg_prob:.4f}")
        print(f"Accuracy (Threshold 0.5): {accuracy:.2f}%")
        print("----------------------\n")

def main(_):
    # Define observation space structure for initialization
    image_keys = ["image_0"]
    
    # Mock observation for initialization
    init_obs = {
        "image_0": np.zeros((1, 1, 128, 128, 3), dtype=np.uint8),
        "state": np.zeros((1, 14), dtype=np.float32)
    }

    rng = jax.random.PRNGKey(0)
    
    print(f"Loading classifier from {FLAGS.checkpoint_path}...")
    classifier_func = load_classifier_func(
        key=rng,
        sample=init_obs,
        image_keys=image_keys,
        checkpoint_path=os.path.abspath(FLAGS.checkpoint_path)
    )
    
    # --- Collect Data ---
    success_paths = []
    failure_paths = []
    
    if not os.path.exists(RECORD_DATA_DIR):
        raise FileNotFoundError(f"Record data dir not found: {RECORD_DATA_DIR}")

    sessions = sorted([d for d in os.listdir(RECORD_DATA_DIR) if os.path.isdir(os.path.join(RECORD_DATA_DIR, d))])
    
    for session in sessions:
        if session in ["success", "failure"]:
            continue
            
        img_dir = os.path.join(RECORD_DATA_DIR, session, "images", "cam_2_rgb")
        if not os.path.exists(img_dir):
            continue
            
        paths = load_images_from_folder(img_dir)
        
        if session == SUCCESS_SESSION:
            success_paths.extend(paths)
        else:
            failure_paths.extend(paths)

    # --- Evaluate ---
    evaluate_images(classifier_func, success_paths, "SUCCESS")
    evaluate_images(classifier_func, failure_paths, "FAILURE")

if __name__ == "__main__":
    app.run(main)
