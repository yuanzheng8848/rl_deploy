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

RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/record_data"

def load_images_from_folder(folder, last_n_frames=10):
    images = []
    # Find all images
    exts = ["*.jpg", "*.png", "*.jpeg"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    
    paths.sort() # Ensure temporal order
    
    if not paths:
        return []
        
    # Take last N frames
    if len(paths) > last_n_frames:
        paths = paths[-last_n_frames:]
        
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        # Expand dims to (1, 1, 128, 128, 3) for Time dimension
        img = img[None, None, ...] 
        images.append(img)
    return images

def evaluate_images(classifier_func, images, label_name):
    print(f"Evaluating {len(images)} images for {label_name}...")
    
    probs = []
    correct_count = 0
    
    for img in tqdm(images):
        # Prepare observation
        obs = {
            "image_0": img,
            "state": np.zeros((1, 14), dtype=np.float32)
        }
        
        # Predict
        logit = classifier_func(obs).item()
        prob = 1 / (1 + np.exp(-logit))
        probs.append(prob)
        
        if label_name == "SUCCESS":
            if prob >= 0.5: correct_count += 1
        else: # FAILURE
            if prob < 0.5: correct_count += 1
            
    avg_prob = np.mean(probs) if probs else 0.0
    accuracy = (correct_count / len(images)) * 100 if images else 0.0
    
    print(f"\n--- {label_name} Results ---")
    print(f"Total Images: {len(images)}")
    print(f"Average Probability: {avg_prob:.4f}")
    print(f"Accuracy (Threshold 0.5): {accuracy:.2f}%")
    print("----------------------\n")

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
    
    # --- Load and Evaluate Success Data ---
    success_dir = os.path.join(RECORD_DATA_DIR, "success")
    success_sessions = sorted([d for d in os.listdir(success_dir) if os.path.isdir(os.path.join(success_dir, d))])
    
    all_success_images = []
    print(f"Loading Success Data from {success_dir}...")
    for session in tqdm(success_sessions):
        cam_dir = os.path.join(success_dir, session, "images", "cam_2_rgb")
        if not os.path.exists(cam_dir):
            continue
        all_success_images.extend(load_images_from_folder(cam_dir, last_n_frames=10))
        
    evaluate_images(classifier_func, all_success_images, "SUCCESS")
    
    # --- Load and Evaluate Failure Data ---
    failure_dir = os.path.join(RECORD_DATA_DIR, "failure")
    failure_sessions = sorted([d for d in os.listdir(failure_dir) if os.path.isdir(os.path.join(failure_dir, d))])
    
    all_failure_images = []
    print(f"Loading Failure Data from {failure_dir}...")
    for session in tqdm(failure_sessions):
        cam_dir = os.path.join(failure_dir, session, "images", "cam_2_rgb")
        if not os.path.exists(cam_dir):
            continue
        all_failure_images.extend(load_images_from_folder(cam_dir, last_n_frames=10))
        
    evaluate_images(classifier_func, all_failure_images, "FAILURE")

if __name__ == "__main__":
    app.run(main)
