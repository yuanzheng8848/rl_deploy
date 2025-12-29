#!/usr/bin/env python3

import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm
from absl import app, flags
from flax.training import checkpoints
import flax.linen as nn
import pickle as pkl
import numpy as np
import os
import copy
import cv2
import glob
import sys
from pathlib import Path

# --- Path Setup ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "serl" / "serl_launcher"))
sys.path.append(str(ROOT_DIR / "serl" / "serl_robot_infra"))
# ------------------

from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

import gym
from gym import spaces

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", "../classifier_ckpt_cam2_last10", "Path to save checkpoint")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")

RECORD_DATA_DIR = "/home/sj/Desktop/zy/moqi_workspace/record_data"

def load_images_from_folder(folder_path, last_n_frames=10):
    images = []
    if not os.path.exists(folder_path):
        return images
    
    # Find all images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_paths.sort() # Ensure temporal order
    
    # Take last N frames
    if len(image_paths) > last_n_frames:
        image_paths = image_paths[-last_n_frames:]
    
    for path in image_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            # Expand dims to (1, 128, 128, 3) for Time dimension
            img = img[None, ...] 
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return images

def populate_data_store_from_images(data_store, images):
    for img in images:
        # Create dummy transition
        transition = {
            "observations": {
                "image_0": img, # (1, 128, 128, 3)
                "state": np.zeros((1, 14), dtype=np.float32)
            },
            "next_observations": {
                "image_0": img,
                "state": np.zeros((1, 14), dtype=np.float32)
            },
            "actions": np.zeros((1, 14), dtype=np.float32),
            "rewards": np.array([0.0], dtype=np.float32), 
            "masks": np.array([1.0], dtype=np.float32),
            "dones": np.array([0.0], dtype=np.float32)
        }
        data_store.insert(transition)
    return data_store

def fix_image_shape(x):
    """
    Ensure image shape is (Batch, 1, 128, 128, 3)
    """
    shape = x.shape
    # If (B, 128, 128, 3), add time dim -> (B, 1, 128, 128, 3)
    if len(shape) == 4 and shape[1] == 128 and shape[2] == 128 and shape[3] == 3:
        x = jnp.expand_dims(x, axis=1)
    
    final_shape = x.shape
    if len(final_shape) != 5 or final_shape[1] != 1 or final_shape[2] != 128 or final_shape[3] != 128 or final_shape[4] != 3:
         # Try to reshape if total elements match
         try:
             x = jnp.reshape(x, (shape[0], 1, 128, 128, 3))
         except:
             # If that fails, maybe it was (B, 128, 128, 3) and we need to expand
             try:
                 x = jnp.reshape(x, (shape[0], 128, 128, 3))
                 x = jnp.expand_dims(x, axis=1)
             except:
                 raise ValueError(f"Could not fix shape {shape} to (B, 1, 128, 128, 3)")
    return x

def main(_):
    STATE_DIM = 14 
    ACTION_DIM = 14 
    image_keys = ["image_0"]
    
    observation_space = spaces.Dict({
        "image_0": spaces.Box(0, 255, shape=(1, 128, 128, 3), dtype=np.uint8),
        "state": spaces.Box(-np.inf, np.inf, shape=(1, STATE_DIM), dtype=np.float32)
    })
    action_space = spaces.Box(-1, 1, shape=(1, ACTION_DIM), dtype=np.float32)

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=100000, image_keys=image_keys
    )
    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space, action_space, capacity=100000, image_keys=image_keys
    )
    
    # --- Load Data ---
    success_images = []
    failure_images = []
    
    if not os.path.exists(RECORD_DATA_DIR):
        raise FileNotFoundError(f"Record data dir not found: {RECORD_DATA_DIR}")

    # Load Success Data
    success_dir = os.path.join(RECORD_DATA_DIR, "success")
    if os.path.exists(success_dir):
        sessions = sorted([d for d in os.listdir(success_dir) if os.path.isdir(os.path.join(success_dir, d))])
        print(f"Loading Success Data from {success_dir}...")
        for session in tqdm(sessions):
            cam_dir = os.path.join(success_dir, session, "images", "cam_2_rgb")
            if not os.path.exists(cam_dir):
                continue
            loaded = load_images_from_folder(cam_dir, last_n_frames=10)
            success_images.extend(loaded)
            
    # Load Failure Data
    failure_dir = os.path.join(RECORD_DATA_DIR, "failure")
    if os.path.exists(failure_dir):
        sessions = sorted([d for d in os.listdir(failure_dir) if os.path.isdir(os.path.join(failure_dir, d))])
        print(f"Loading Failure Data from {failure_dir}...")
        for session in tqdm(sessions):
            cam_dir = os.path.join(failure_dir, session, "images", "cam_2_rgb")
            if not os.path.exists(cam_dir):
                continue
            loaded = load_images_from_folder(cam_dir, last_n_frames=10)
            failure_images.extend(loaded)

    print(f"Total Success Images: {len(success_images)}")
    print(f"Total Failure Images: {len(failure_images)}")

    if not success_images:
        raise ValueError("No success images found.")
    if not failure_images:
        raise ValueError("No failure images found.")

    populate_data_store_from_images(pos_buffer, success_images)
    populate_data_store_from_images(neg_buffer, failure_images)

    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(devices, ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    
    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False}, 
        device=sharding
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2, "pack_obs_and_next_obs": False}, 
        device=sharding
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    
    # --- Init Network ---
    init_batch = next(pos_iterator)
    init_obs_processed = {}
    for k in image_keys:
        # 获取 Batch 数据
        batch_data = fix_image_shape(init_batch["observations"][k])
        # 只取第一条数据 (Single Sample) 用于初始化 -> (1, 128, 128, 3)
        init_obs_processed[k] = batch_data[0]

    print(f"Init sample shape: {init_obs_processed['image_0'].shape}") 
    classifier = create_classifier(key, init_obs_processed, image_keys)
    # --------------------

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch["data"], rngs={"dropout": key}, train=True)
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn({"params": state.params}, batch["data"], train=False, rngs={"dropout": key})
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return state.apply_gradients(grads=grads), loss, train_accuracy

    print("Starting training...")
    for epoch in tqdm(range(FLAGS.num_epochs)):
        try:
            pos_sample = next(pos_iterator)
            neg_sample = next(neg_iterator)
        except StopIteration:
            continue

        def process_obs_batch(obs_dict):
            new_obs = {}
            for k in image_keys:
                new_obs[k] = fix_image_shape(obs_dict[k])
            return new_obs

        pos_obs = process_obs_batch(pos_sample["observations"])
        neg_obs = process_obs_batch(neg_sample["observations"])

        sample = concat_batches(pos_obs, neg_obs, axis=0)
        
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)

        labels = jnp.concatenate([
            jnp.ones((FLAGS.batch_size // 2, 1)), 
            jnp.zeros((FLAGS.batch_size // 2, 1))
        ], axis=0)
        
        batch = {"data": sample, "labels": labels}

        rng, key = jax.random.split(rng)
        classifier, loss, acc = train_step(classifier, batch, key)
        
        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: Loss {loss:.4f}, Acc {acc:.4f}")

    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    
    abs_checkpoint_path = os.path.abspath(FLAGS.checkpoint_path)
    checkpoints.save_checkpoint(abs_checkpoint_path, classifier, step=FLAGS.num_epochs, overwrite=True)
    print(f"Classifier saved to {abs_checkpoint_path}")

if __name__ == "__main__":
    app.run(main)
