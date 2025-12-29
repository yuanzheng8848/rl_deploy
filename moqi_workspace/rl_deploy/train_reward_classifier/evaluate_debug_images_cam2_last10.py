#!/usr/bin/env python3

import glob
import tqdm
import cv2
import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from tqdm import tqdm
import sys
import os
from pathlib import Path

# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "serl" / "serl_launcher"))
sys.path.append(str(ROOT_DIR / "serl" / "serl_robot_infra"))
# ------------------

from serl_launcher.networks.reward_classifier import load_classifier_func

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "../classifier_ckpt_cam2_last10", "Path to the classifier checkpoint")
flags.DEFINE_string("debug_images_dir", "../debug_classifier_images", "Path to debug images")

def load_and_process_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        return None
    
    # Debug images are saved as BGR on disk (cv2 default).
    # We need to convert to RGB for the classifier.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    # Add batch and time dimensions: (1, 1, 128, 128, 3)
    img = img[None, None, ...]
    return img

def main(_):
    # Initialize Classifier
    image_keys = ["image_0"]
    init_obs = {
        "image_0": np.zeros((1, 1, 128, 128, 3), dtype=np.uint8),
        "state": np.zeros((1, 14), dtype=np.float32)
    }
    rng = jax.random.PRNGKey(0)
    
    ckpt_path = os.path.abspath(FLAGS.checkpoint_path)
    print(f"Loading classifier from {ckpt_path}...")
    classifier_func = load_classifier_func(
        key=rng,
        sample=init_obs,
        image_keys=image_keys,
        checkpoint_path=ckpt_path
    )
    
    debug_dir = os.path.abspath(FLAGS.debug_images_dir)
    if not os.path.exists(debug_dir):
        print(f"Debug directory not found: {debug_dir}")
        return

    # Find images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(debug_dir, ext)))
    
    image_paths.sort()
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Evaluating {len(image_paths)} images from {debug_dir}...")
    
    print("\nFilename | Probability")
    print("--- | ---")
    
    probs = []
    
    for path in image_paths:
        try:
            img = load_and_process_image(path)
            if img is None:
                continue
            
            input_obs = {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32)
            }
            
            logit = classifier_func(input_obs).item()
            prob = 1 / (1 + np.exp(-logit))
            probs.append(prob)
            
            filename = os.path.basename(path)
            print(f"{filename} | {prob:.4f}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if probs:
        print(f"\nAverage Probability: {np.mean(probs):.4f}")
        print(f"Min Probability: {np.min(probs):.4f}")
        print(f"Max Probability: {np.max(probs):.4f}")

if __name__ == "__main__":
    app.run(main)
