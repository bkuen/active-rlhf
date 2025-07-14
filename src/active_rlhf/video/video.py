import imageio
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import Env


def render_observation_frame(env: Env, obs: th.Tensor):
    mujoco_env = env.unwrapped
    is_mujoco = hasattr(mujoco_env, 'set_state') and hasattr(mujoco_env, 'data')

    frames = []
    if is_mujoco:
        # Dynamically determine qpos/qvel split
        model = getattr(mujoco_env, 'model', None)
        if model is not None and hasattr(model, 'nq') and hasattr(model, 'nv'):
            nq = model.nq
            nv = model.nv
        else:
            # Fallback: try to infer from data shapes
            nq = mujoco_env.data.qpos.shape[0]
            nv = mujoco_env.data.qvel.shape[0]
        for i in range(len(obs)):
            obs_i = obs[i].cpu().numpy() if hasattr(obs[i], 'cpu') else np.array(obs[i])
            qpos = mujoco_env.data.qpos.copy()
            qvel = mujoco_env.data.qvel.copy()
            # Try to fill qpos and qvel from obs
            if obs_i.shape[0] == nq + nv - 1:
                # Some envs (like HalfCheetah) hide root x (qpos[0])
                qpos[1:] = obs_i[:nq - 1]
                qvel[:] = obs_i[nq - 1:]
            elif obs_i.shape[0] == nq + nv:
                # Full state in obs
                qpos[:] = obs_i[:nq]
                qvel[:] = obs_i[nq:]
            else:
                # Fallback: try to fill as much as possible
                qpos[:min(nq, obs_i.shape[0])] = obs_i[:min(nq, obs_i.shape[0])]
            mujoco_env.set_state(qpos, qvel)
            frame = env.render()
            frames.append(frame)
    else:
        # Non-MuJoCo: just reset and render each obs (best effort)
        print("Warning: Environment does not support MuJoCo-style state setting. Rendering resets only.")
        for i in range(len(obs)):
            env.reset()
            frame = env.render()
            frames.append(frame)

    return frames


def render_observation(obs: th.Tensor, save_path: str, env_id: str):
    """
    Render a sequence of observations as a video, supporting MuJoCo and non-MuJoCo environments.
    For MuJoCo envs, sets the state using qpos/qvel. For others, just resets and renders.
    """
    import numpy as np
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    frames = render_observation_frame(env, obs)

    # Save frames as a video
    imageio.mimsave(save_path, frames, fps=30, codec="libx264")


def render_observation_comparison(original_obs: th.Tensor, recon_obs: th.Tensor, save_path: str, env_id: str,
                                  num_samples: int = 5):
    """
    Render a comparison video showing original vs reconstructed observations for multiple samples.
    Creates a video with num_samples rows, each showing original (left) vs reconstructed (right).
    """
    # Limit number of samples to available data
    num_samples = min(num_samples, original_obs.shape[0], recon_obs.shape[0])

    env = gym.make(env_id, render_mode="rgb_array")
    env.reset()

    # Get trajectory length
    traj_length = original_obs.shape[1]

    # For each timestep, create a frame showing all samples side by side
    frames = []
    for t in range(traj_length):
        # Create a list to store frames for each sample
        sample_frames = []

        for sample_idx in range(num_samples):
            # Get original and reconstructed observations for this timestep
            orig_obs_t = original_obs[sample_idx, t].unsqueeze(0)  # Add batch dimension
            recon_obs_t = recon_obs[sample_idx, t].unsqueeze(0)  # Add batch dimension

            # Render original observation
            orig_frames = render_observation_frame(env, orig_obs_t)
            orig_frame = orig_frames[0] if orig_frames else env.render()

            # Render reconstructed observation
            recon_frames = render_observation_frame(env, recon_obs_t)
            recon_frame = recon_frames[0] if recon_frames else env.render()

            # Concatenate original and reconstructed frames horizontally
            combined_frame = np.concatenate([orig_frame, recon_frame], axis=1)
            sample_frames.append(combined_frame)

        # Concatenate all samples vertically
        if sample_frames:
            combined_frame = np.concatenate(sample_frames, axis=0)
            frames.append(combined_frame)

    # Save frames as a video
    if frames:
        imageio.mimsave(save_path, frames, fps=30, codec="libx264")
        print(f"Comparison video saved to {save_path}")
    else:
        print("No frames generated for comparison video")

    env.close()