#!/usr/bin/env python3

import os
import sys
import ctypes

# --- Force NVIDIA Library Paths for JAX ---
# This must be done BEFORE importing jax or any other library that might use it
nvidia_base = "/home/peter/miniconda3/envs/zy/lib/python3.10/site-packages/nvidia"
libs = [
    "cublas/lib", "cudnn/lib", "cufft/lib", "cusolver/lib", 
    "cusparse/lib", "nccl/lib", "nvjitlink/lib"
]
for lib in libs:
    path = os.path.join(nvidia_base, lib)
    if os.path.exists(path):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld}"

# Set XLA_FLAGS to help JAX find CUDA
os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={nvidia_base}"

# Explicitly preload libraries in dependency order
try:
    # Preload nvJitLink FIRST (cusparse depends on it)
    nvjitlink_path = os.path.join(nvidia_base, "nvjitlink/lib/libnvJitLink.so.12")
    if os.path.exists(nvjitlink_path):
        ctypes.CDLL(nvjitlink_path)
        print(f"[DEBUG] Successfully preloaded {nvjitlink_path}")
    
    # Then preload cuSPARSE
    cusparse_path = os.path.join(nvidia_base, "cusparse/lib/libcusparse.so.12")
    if os.path.exists(cusparse_path):
        ctypes.CDLL(cusparse_path)
        print(f"[DEBUG] Successfully preloaded {cusparse_path}")
        
    sys.stdout.flush()
except Exception as e:
    print(f"[DEBUG] Failed to preload libraries: {e}")
    sys.stdout.flush()
# ------------------------------------------

import time
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import pickle as pkl
import os
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics


# --- Path Setup ---
# Add serl and pyroki to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR / "serl" / "serl_launcher"))
sys.path.append(str(ROOT_DIR / "serl" / "serl_robot_infra"))
sys.path.append(str(ROOT_DIR / "moqi_workspace" / "pyroki"))

# Import OpenArmEnv
from openarm_env import OpenArmEnv

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func
from franka_env.envs.wrappers import BinaryRewardClassifierWrapper, Quat2EulerWrapper
from franka_env.envs.relative_env import RelativeFrame

class DeployBinaryRewardWrapper(gym.Wrapper):
    """
    Compute binary reward with custom classifier fn: reward = 1 if prob > 0.5 else 0
    """
    def __init__(self, env, reward_classifier_func, reward_classifier_func_cam1=None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.reward_classifier_func_cam1 = reward_classifier_func_cam1

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        
        prob = 0.0
        if self.reward_classifier_func is not None:
            # Create a copy of obs to modify for classifier
            temp_obs = {}
            # Map 'image_primary' (from Env) to 'image_0' (expected by Classifier Checkpoint)
            if "image_primary" in obs:
                 img = obs["image_primary"]
                 # Ensure images have (Batch, Time) dimensions -> (1, 1, H, W, C)
                 if img.ndim == 3: # (H, W, C)
                      img = img[None, None, ...]
                 elif img.ndim == 4: # (Time, H, W, C) -> (1, Time, H, W, C)
                      img = img[None, ...] 
                 
                 temp_obs["image_0"] = img
            else:
                 temp_obs = obs.copy()

            # ---------------------
            # ---------------------

            logit = self.reward_classifier_func(temp_obs).item()
            prob_primary = 1 / (1 + np.exp(-logit))
            
            final_prob = prob_primary
            
            # --- Cam1 Classifier ---
            if self.reward_classifier_func_cam1 is not None:
                temp_obs_cam1 = {}
                # Map 'image_right' (cam1) to 'image_0'
                if "image_right" in obs:
                     img_cam1 = obs["image_right"]
                     if img_cam1.ndim == 3:
                          img_cam1 = img_cam1[None, None, ...]
                     elif img_cam1.ndim == 4:
                          img_cam1 = img_cam1[None, ...]
                     
                     temp_obs_cam1["image_0"] = img_cam1
                else:
                     # Fallback or error? For now fallback to copy but it likely won't have image_0 correct if not present
                     temp_obs_cam1 = obs.copy()
                
                if "image_0" in temp_obs_cam1:
                    logit_cam1 = self.reward_classifier_func_cam1(temp_obs_cam1).item()
                    prob_cam1 = 1 / (1 + np.exp(-logit_cam1))
                    
                    # Average probabilities for Reward
                    final_prob = (prob_primary + prob_cam1) / 2.0
                    info["classifier_prob_cam1"] = prob_cam1
            # -----------------------
            
        # Binary Reward
        binary_reward = 1.0 if final_prob > 0.35 else 0.0
        rew += binary_reward
        
        # Terminate if highly successful (using combined prob)
        if final_prob > 0.7:
            done = True
            
        info["classifier_prob"] = prob_primary # Store RAW primary prob for visualization
        info["combined_prob"] = final_prob
        return obs, rew, done, truncated, info


FLAGS = flags.FLAGS

flags.DEFINE_string("env", "OpenArm-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", "openarm_rl_deploy", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 20, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 100000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 50000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 10, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 10, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", True, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", "./train_reward_classifier/rl_success_demos.pkl", "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 1000, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", os.path.abspath("./rl_ckpt"), "Path to save checkpoints.")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", os.path.abspath("./classifier_ckpt_cam2_last10/checkpoint_100"), "Path to reward classifier ckpt."
)
flags.DEFINE_string(
    "reward_classifier_ckpt_path_cam1", os.path.abspath("./classifier_ckpt_cam1/checkpoint_100"), "Path to reward classifier ckpt for cam1."
)

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: DrQAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            data_store.insert(transition)

            obs = next_obs
            
            # --- Visualization ---
            try:
                if FLAGS.render:
                    import cv2
                    # Extract image (handle ChunkingWrapper adding dim)
                    # Assuming image_primary is available
                    if "image_primary" in next_obs:
                        vis_img = next_obs["image_primary"]
                        if vis_img.ndim == 4: # (1, 128, 128, 3)
                            vis_img = vis_img[0]
                        
                        # Convert RGB to BGR for OpenCV
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                        
                        # Resize for better visibility
                        vis_img = cv2.resize(vis_img, (512, 512))
                        
                        # Draw Reward
                        reward_text = f"Reward: {reward:.2f}"
                        # Default color for reward text
                        reward_color = (0, 255, 0) if reward > 0.5 else (0, 0, 255)
                        cv2.putText(vis_img, reward_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, reward_color, 2)

                        # info is 0-d array wrapping dict
                        info_dict = info.item() if isinstance(info, np.ndarray) and info.ndim == 0 else info
                        
                        if isinstance(info_dict, dict) and "classifier_prob" in info_dict:
                            prob = info_dict["classifier_prob"]
                            is_success = prob >= 0.5
                            prob_color = (0, 255, 0) if is_success else (0, 0, 255)
                            prob_text = f"Prob: {prob:.3f}"
                            # Draw Prob on a new line (y=60)
                            cv2.putText(vis_img, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)

                        # --- Cam1 Visualization ---
                        if "image_right" in next_obs:
                            vis_img_cam1 = next_obs["image_right"]
                            if vis_img_cam1.ndim == 4:
                                vis_img_cam1 = vis_img_cam1[0]
                            vis_img_cam1 = cv2.cvtColor(vis_img_cam1, cv2.COLOR_RGB2BGR)
                            vis_img_cam1 = cv2.resize(vis_img_cam1, (512, 512))
                            
                            if isinstance(info_dict, dict) and "classifier_prob_cam1" in info_dict:
                                prob_cam1 = info_dict["classifier_prob_cam1"]
                                prob_text_cam1 = f"Prob1: {prob_cam1:.3f}"
                                color_cam1 = (0, 255, 0) if prob_cam1 >= 0.5 else (0, 0, 255)
                                cv2.putText(vis_img_cam1, prob_text_cam1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cam1, 2)
                            
                            # Concatenate horizontally
                            vis_img = np.concatenate([vis_img, vis_img_cam1], axis=1)
                        # --------------------------

                        # --- Q-Value Visualization ---
                        # Query critic for current and perturbed states
                        # Perturbations: None, +X, -X, +Y, -Y, +Z, -Z
                        perturbations = [
                            ("Curr", [0.0, 0.0, 0.0]),
                            ("+X",   [0.02, 0.0, 0.0]),
                            ("-X",   [-0.02, 0.0, 0.0]),
                            ("+Y",   [0.0, 0.02, 0.0]),
                            ("-Y",   [0.0, -0.02, 0.0]),
                            ("+Z",   [0.0, 0.0, 0.02]),
                            ("-Z",   [0.0, 0.0, -0.02]),
                        ]
                        
                        # Prepare batch of observations
                        # obs is a dict of arrays. We need to stack them.
                        # obs["state"] shape is likely (1, 13) or (13,)
                        # We assume ChunkingWrapper is used, so (1, 13).
                        
                        batch_obs = {}
                        for k, v in obs.items():
                            # v shape: (Time, ...) e.g. (1, 13)
                            # Add batch dim: (1, Time, ...)
                            v_batched = v[None, ...]
                            # Tile: (Batch, Time, ...)
                            tile_arg = (len(perturbations),) + (1,) * v.ndim
                            batch_obs[k] = np.tile(v_batched, tile_arg)
                            
                        # Apply perturbations to "state"
                        # Assuming state index 0, 1, 2 are X, Y, Z
                        if "state" in batch_obs:
                            for i, (_, offset) in enumerate(perturbations):
                                # batch_obs["state"] shape: (N, 1, 13)
                                batch_obs["state"][i, 0, 0] += offset[0]
                                batch_obs["state"][i, 0, 1] += offset[1]
                                batch_obs["state"][i, 0, 2] += offset[2]
                        
                        # Prepare batch of actions
                        # actions shape: (1, 7) or (7,)
                        # If actions is (7,), expand to (1, 7)
                        actions_batch = actions
                        if actions_batch.ndim == 1:
                            actions_batch = actions_batch[None, ...]
                        actions_batch = np.tile(actions_batch, (len(perturbations), 1))
                        
                        # Query Critic
                        # We need a key for random number generator, though we are not training (dropout)
                        # We can use a dummy key or split sampling_rng (but we are in a loop, be careful with state)
                        # agent.forward_critic expects jax arrays
                        
                        # Use a temporary key for visualization to not affect main training loop RNG if possible,
                        # or just split off a new one.
                        # We don't have access to `sampling_rng` easily here without modifying the loop variable.
                        # But `actor` function has `sampling_rng`.
                        # Let's just use a new PRNGKey for visualization to be safe and stateless w.r.t training loop
                        vis_rng = jax.random.PRNGKey(0) 
                        
                        q_values = agent.forward_critic(
                            jax.device_put(batch_obs),
                            jax.device_put(actions_batch),
                            rng=vis_rng,
                            train=False
                        )
                        # q_values shape: (ensemble_size, batch_size)
                        # Take mean over ensemble
                        q_values_mean = jnp.mean(q_values, axis=0)
                        q_values_np = np.asarray(q_values_mean)
                        
                        # Display Q-values (Bar Chart)
                        # Create a separate canvas for Q-values
                        chart_w, chart_h = 400, 512
                        chart_img = np.zeros((chart_h, chart_w, 3), dtype=np.uint8)
                        
                        # Find min/max for scaling
                        q_min, q_max = q_values_np.min(), q_values_np.max()
                        q_range = q_max - q_min
                        if q_range < 1e-6: q_range = 1.0
                        
                        # Margins
                        margin_x = 40
                        margin_y = 40
                        bar_width = (chart_w - 2 * margin_x) // len(perturbations)
                        
                        # Draw bars
                        curr_val = q_values_np[0]
                        for i, (label, _) in enumerate(perturbations):
                            val = q_values_np[i]
                            
                            # Normalize height (0 to 1) relative to min/max of the set
                            # Or better: center around current value?
                            # Let's just map min-max to 10%-90% of height
                            norm_h = (val - q_min) / q_range
                            bar_h = int(norm_h * (chart_h - 2 * margin_y))
                            
                            # Coordinates
                            x1 = margin_x + i * bar_width + 5
                            x2 = x1 + bar_width - 10
                            y2 = chart_h - margin_y
                            y1 = y2 - bar_h
                            
                            # Color
                            if i == 0: color = (255, 255, 0) # Cyan for Current
                            elif val > curr_val: color = (0, 255, 0) # Green for better
                            else: color = (0, 0, 255) # Red for worse
                            
                            cv2.rectangle(chart_img, (x1, y1), (x2, y2), color, -1)
                            
                            # Text Label
                            cv2.putText(chart_img, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(chart_img, f"{val:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Concatenate chart to the right
                        vis_img = np.concatenate([vis_img, chart_img], axis=1)
                        
                        cv2.imshow("Actor View", vis_img)
                        cv2.waitKey(1)
            except Exception as e:
                print(f"Vis Error: {e}")
            # ---------------------

            if done or truncated:
                if reward:
                    # Check if it's a real success based on classifier probability
                    is_success = False
                    # info is 0-d array wrapping dict
                    info_dict = info.item() if isinstance(info, np.ndarray) and info.ndim == 0 else info
                    
                    if isinstance(info_dict, dict) and "classifier_prob" in info_dict:
                        if info_dict["classifier_prob"] > 0.95:
                            is_success = True
                    
                    if is_success:
                        print("OpenArm task success!")
                stats = {"train": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, demo_buffer):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    # Load checkpoint if it exists
    if os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
        if latest_ckpt:
            # Restore agent state
            agent = agent.replace(state=checkpoints.restore_checkpoint(FLAGS.checkpoint_path, agent.state))
            # Restore update_steps from the checkpoint path (assuming format "checkpoint_X")
            try:
                update_steps = int(latest_ckpt.split("_")[-1])
                print_green(f"Restored checkpoint from {latest_ckpt} at step {update_steps}")
            except ValueError:
                print(f"Warning: Could not parse step from checkpoint path {latest_ckpt}")


    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(
                    batch,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            # Convert JAX arrays to numpy for WandB logging
            update_info_np = jax.device_get(update_info)
            wandb_logger.log(update_info_np, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100, overwrite=True
            )

        update_steps += 1


##############################################################################


flags.DEFINE_string("arm", "both", "Which arm to control: 'left', 'right', or 'both'.")

# ... (existing flags) ...

def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # create env and load dataset
    # Instantiate OpenArmEnv directly
    from openarm_env import DefaultOpenArmConfig
    config = DefaultOpenArmConfig()
    # Add a camera to ensure image_keys is not empty
    # The serial number is a placeholder if using fake_env, or needs to be real if using real env
    # For now, we assume fake_env or that the user will configure the real serials
    config.REALSENSE_CAMERAS = {
        "image_primary": "248622302807",
        "image_left": "150622074105",
        "image_right": "236422072385"
    } 
    
    env = OpenArmEnv(
        fake_env=FLAGS.learner, # Learner uses fake env (no hardware connection)
        save_video=FLAGS.eval_checkpoint_step,
        config=config,
        max_episode_length=FLAGS.max_traj_length,
        arm=FLAGS.arm,
    )
    
    # Wrappers
    # Use RelativeFrame to make policy robust to absolute position shifts
    env = RelativeFrame(env)
    # Use Quat2Euler to simplify rotation representation (optional but recommended)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    
    if FLAGS.actor:
        # initialize the classifier and wrap the env
        if FLAGS.reward_classifier_ckpt_path is None:
            raise ValueError("reward_classifier_ckpt_path must be specified for actor")

        # The classifier was likely trained on a single image (image_0/image_primary).
        # We must restrict the input to just that image to avoid shape mismatch (768 vs 256).
        # IMPORTANT: The checkpoint was trained with "image_0". We MUST use "image_0" here
        # so that the parameter names (encoder_image_0) match the checkpoint.
        reward_image_keys = ["image_0"]
        
        # IMPORTANT: The checkpoint was trained with "image_0". We MUST use "image_0" here
        # so that the parameter names (encoder_image_0) match the checkpoint.
        reward_image_keys = ["image_0"]
        
        # Create a sample input that matches the expected keys (image_0)
        # We take a real sample from env and rename the key
        env_sample = env.observation_space.sample()
        classifier_sample = env_sample.copy()
        if "image_primary" in env_sample:
            # We need to make sure the shape matches what the classifier expects if there's any batch dim logic, 
            # but usually sample() returns (H, W, C). 
            # The classifier init will process this. 
            # NOTE: If classifier expects (1, 1, 128, 128, 3) during init, we might need to expand dims here too?
            # Creating a dummy array with correct shape is safer.
            classifier_sample["image_0"] = env_sample["image_primary"]
            
        reward_func = load_classifier_func(
            key=sampling_rng,
            sample=classifier_sample,
            image_keys=reward_image_keys,
            checkpoint_path=FLAGS.reward_classifier_ckpt_path,
        )
        
        reward_func_cam1 = None
        if FLAGS.reward_classifier_ckpt_path_cam1:
             print(f"Loading Cam1 Classifier from {FLAGS.reward_classifier_ckpt_path_cam1}...")
             # Prepare sample for cam1 (image_right -> image_0)
             classifier_sample_cam1 = env_sample.copy()
             if "image_right" in env_sample:
                 classifier_sample_cam1["image_0"] = env_sample["image_right"]
             
             reward_func_cam1 = load_classifier_func(
                key=sampling_rng,
                sample=classifier_sample_cam1,
                image_keys=reward_image_keys,
                checkpoint_path=FLAGS.reward_classifier_ckpt_path_cam1,
            )
            
        env = DeployBinaryRewardWrapper(env, reward_func, reward_func_cam1)
    
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    # replicate agent across devices
    agent: DrQAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=10000,
            image_keys=image_keys,
        )

        if FLAGS.demo_path:
            # Check if the file exists
            if not os.path.exists(FLAGS.demo_path):
                raise FileNotFoundError(f"File {FLAGS.demo_path} not found")

            with open(FLAGS.demo_path, "rb") as f:
                trajs = pkl.load(f)
                for traj in trajs:
                    # traj is a dict of arrays, we need to insert individual transitions
                    # Assuming all keys have the same length
                    traj_len = len(traj["actions"])
                    for i in range(traj_len):
                        # Slice state and actions based on FLAGS.arm
                        # Assuming demo data is always dual arm (14 dim action, 2x7+2x1 state)
                        
                        # --- Adapt for RelativeFrame and Quat2EulerWrapper ---
                        # 1. Get Reset Pose (First frame of trajectory)
                        # Assuming demo data structure: state is (N, 16) or similar
                        # We need to extract the 7-dim pose (XYZ + Quat)
                        
                        # Helper to get pose from state vector
                        def get_pose(s, arm):
                            if arm == "left": return s[:7]
                            elif arm == "right": return s[8:15] # 8-14 is pose (7), 15 is gripper
                            return s[:7] # Default
                            
                        def get_gripper(s, arm):
                            if arm == "left": return s[7:8]
                            elif arm == "right": return s[15:16]
                            return s[7:8]

                        # Get initial pose (Reset Pose) for this trajectory
                        # We assume the first step i=0 is the reset pose, but we are iterating i.
                        # We need to access the 0-th element of the WHOLE trajectory arrays.
                        # traj["observations"]["state"] is the whole array.
                        
                        reset_state_vec = traj["observations"]["state"][0]
                        reset_pose_abs = get_pose(reset_state_vec, FLAGS.arm)
                        
                        # Calculate T_reset_inv
                        from franka_env.utils.transformations import construct_homogeneous_matrix, construct_adjoint_matrix
                        from scipy.spatial.transform import Rotation as R
                        
                        T_reset = construct_homogeneous_matrix(reset_pose_abs)
                        T_reset_inv = np.linalg.inv(T_reset)
                        
                        # Process current step
                        curr_state_vec = traj["observations"]["state"][i]
                        next_state_vec = traj["next_observations"]["state"][i]
                        
                        curr_pose_abs = get_pose(curr_state_vec, FLAGS.arm)
                        next_pose_abs = get_pose(next_state_vec, FLAGS.arm)
                        
                        curr_gripper = get_gripper(curr_state_vec, FLAGS.arm)
                        next_gripper = get_gripper(next_state_vec, FLAGS.arm)
                        
                        # 2. Compute Relative Pose
                        def to_relative(pose_abs, T_inv):
                            T_curr = construct_homogeneous_matrix(pose_abs)
                            T_rel = T_inv @ T_curr
                            p_rel = T_rel[:3, 3]
                            q_rel = R.from_matrix(T_rel[:3, :3]).as_quat()
                            return np.concatenate([p_rel, q_rel])
                            
                        curr_pose_rel = to_relative(curr_pose_abs, T_reset_inv)
                        next_pose_rel = to_relative(next_pose_abs, T_reset_inv)
                        
                        # 3. Convert to Euler (Quat2EulerWrapper)
                        def to_euler(pose_quat):
                            # pose_quat is (7,) [xyz, qx, qy, qz, qw]
                            xyz = pose_quat[:3]
                            quat = pose_quat[3:]
                            rpy = R.from_quat(quat).as_euler("xyz")
                            return np.concatenate([xyz, rpy])
                            
                        curr_pose_euler = to_euler(curr_pose_rel) # (6,)
                        next_pose_euler = to_euler(next_pose_rel) # (6,)
                        
                        # 4. Add Dummy Velocity (6,)
                        vel = np.zeros((6,), dtype=np.float32)
                        
                        # 5. Construct Final State Dictionary
                        # Note: The buffer expects a DICT for observations if env.observation_space is Dict.
                        # But here we are constructing the "state" key's content.
                        # Wait, MemoryEfficientReplayBufferDataStore structure matches env.observation_space.
                        # env.observation_space["state"] is a Dict? 
                        # Let's check OpenArmEnv. It returns {"state": {"tcp_pose": ..., "tcp_vel": ..., "gripper_pose": ...}}
                        # So we should construct that dictionary structure.
                        
                        # HOWEVER, the demo data `traj["observations"]` might be a flat dict or nested.
                        # The existing code `traj["observations"]["state"]` implies "state" is a key.
                        # But `demo_buffer.insert` expects the structure to match.
                        
                        # Let's look at how `transition` is built below.
                        # It puts `state` as a key in `observations`.
                        # But `OpenArmEnv` returns `state` as a DICT containing tcp_pose, etc.
                        # The `SERLObsWrapper` might flatten it?
                        # Let's check `SERLObsWrapper`.
                        # Usually SERLObsWrapper flattens "state" dict into a single vector if configured.
                        # But here `OpenArmEnv` returns a dict for "state".
                        # If `SERLObsWrapper` is used, it usually expects `state` to be a vector or handles dicts.
                        # Wait, `OpenArmEnv` BEFORE my change returned `state` as a dict: `{"tcp_pose": ..., "gripper_pose": ...}`
                        # BUT `_get_obs` returns `{"state": state_observation}`.
                        # So `obs["state"]` IS a dict.
                        
                        # BUT the error message `ValueError: could not broadcast input array from shape (8,) into shape (1,13)`
                        # implies that `state` is treated as a single array of size 13.
                        # This happens if `SERLObsWrapper` or `ChunkingWrapper` flattens the state.
                        # `SERLObsWrapper` typically flattens the `state` dict into a single vector `state`.
                        
                        # So we need to construct a single vector of size 13: [pose(6), vel(6), gripper(1)].
                        
                        final_state = np.concatenate([curr_pose_euler, vel, curr_gripper]) # (13,)
                        final_next_state = np.concatenate([next_pose_euler, vel, next_gripper]) # (13,)
                        
                        # Action Transformation (Adjoint)
                        # RelativeFrame transforms action from Body to Base.
                        # The demo actions are already in Base frame (absolute).
                        # We need to transform them to Body frame (inverse of RelativeFrame.transform_action).
                        # action_body = adjoint_inv @ action_base
                        
                        T_curr_abs = construct_homogeneous_matrix(curr_pose_abs)
                        adjoint = construct_adjoint_matrix(curr_pose_abs) # Adjoint of ABSOLUTE pose
                        adjoint_inv = np.linalg.inv(adjoint)
                        
                        # Slice action based on arm
                        action = traj["actions"][i]
                        if FLAGS.arm == "left":
                            arm_action = action[:7]
                        elif FLAGS.arm == "right":
                            arm_action = action[7:]
                        else:
                            arm_action = action[:7] # Default
                            
                        action_base = arm_action[:6] # XYZ + RPY
                        gripper_action = arm_action[6:] # Gripper
                        
                        action_body_6d = adjoint_inv @ action_base
                        final_action = np.concatenate([action_body_6d, gripper_action]) # (7,)
                        
                        # --- Reward Shaping ---
                        # Success Demos: Last 10 steps are success (0.5), others are path (0.0)
                        # Failure Demos: All 0.0 (or -0.5? But we only have success demos here)
                        # We assume rl_success_demos.pkl ONLY contains success demos.
                        
                        is_success_step = i >= (traj_len - 10)
                        reward_val = 0.5 if is_success_step else 0.0
                        
                        transition = {
                            "observations": {
                                "state": final_state,
                                **{k: v[i] for k, v in traj["observations"].items() if k != "state"}
                            },
                            "next_observations": {
                                "state": final_next_state,
                                **{k: v[i] for k, v in traj["next_observations"].items() if k != "state"}
                            },
                            "actions": final_action,
                            "rewards": np.array(reward_val, dtype=np.float32), # Overwrite reward
                            "masks": np.array(1.0, dtype=np.float32), # Ensure mask is 1.0
                            "dones": traj["dones"][i],
                        }
                        demo_buffer.insert(transition)
            print(f"demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor
        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
