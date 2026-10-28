"""
scripts/train.py
Training execution loop for Multi-Agent PPO (MAPPO) controlling satellite beam hopping.
"""

import sys
import torch
import logging
import numpy as np
from pathlib import Path

# Ensure 'src' is in Python path when executing from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.envs.sat_env import SatComBeamEnv
from src.models.mappo_policy import MAPPOPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_mappo")

# Hyperparameters
NUM_EPISODES = 100
EPISODE_LENGTH = 50
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
CHECKPOINT_DIR = Path("checkpoints")


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
            
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
        
    returns = advantages + values
    return advantages, returns


def main():
    logger.info("Initializing MAPPO Satellite Beam-Hopping Training Run...")

    # 1. Environment Setup
    num_sats = 4
    num_cells = 8
    env = SatComBeamEnv(num_satellites=num_sats, num_cells=num_cells)
    
    # Dimensions: obs_dim per agent, act_dim per agent, global_state_dim for centralized critic
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    global_state_dim = obs_dim * num_sats

    # 2. MAPPO Policy Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using execution device: {device}")

    policy = MAPPOPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        global_state_dim=global_state_dim,
        lr=LEARNING_RATE,
        device=device
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Training Loop
    for episode in range(1, NUM_EPISODES + 1):
        obs, global_state = env.reset()
        
        ep_obs, ep_actions, ep_log_probs = [], [], []
        ep_rewards, ep_values, ep_dones, ep_global_states = [], [], [], []
        total_reward = 0.0

        for step in range(EPISODE_LENGTH):
            # Tensor conversions
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
            global_state_tensor = torch.tensor(global_state, dtype=torch.float32, device=device)

            # Get actions and values from policy
            with torch.no_grad():
                actions, log_probs, values = policy.get_actions(obs_tensor, global_state_tensor)

            # Environment step
            next_obs, next_global_state, rewards, done, _ = env.step(actions.cpu().numpy())

            # Buffer storing
            ep_obs.append(obs_tensor)
            ep_actions.append(actions)
            ep_log_probs.append(log_probs)
            ep_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            ep_values.append(values)
            ep_dones.append(torch.tensor(done, dtype=torch.float32, device=device))
            ep_global_states.append(global_state_tensor)

            total_reward += sum(rewards)
            obs = next_obs
            global_state = next_global_state

            if done:
                break

        # Compute next value for GAE boundary
        with torch.no_grad():
            next_global_tensor = torch.tensor(global_state, dtype=torch.float32, device=device)
            next_value = policy.get_value(next_global_tensor)

        # Convert buffers to tensors
        b_obs = torch.stack(ep_obs)
        b_actions = torch.stack(ep_actions)
        b_log_probs = torch.stack(ep_log_probs)
        b_rewards = torch.stack(ep_rewards)
        b_values = torch.stack(ep_values)
        b_dones = torch.stack(ep_dones)
        b_global_states = torch.stack(ep_global_states)

        # Calculate Advantages & Returns
        advantages, returns = compute_gae(
            b_rewards, b_values, next_value, b_dones, gamma=GAMMA, lam=GAE_LAMBDA
        )

        # Update MAPPO Policy
        actor_loss, critic_loss = policy.update(
            b_obs, b_global_states, b_actions, b_log_probs, returns, advantages, epochs=PPO_EPOCHS
        )

        # Logging
        if episode % 10 == 0 or episode == 1:
            logger.info(
                f"Episode {episode:3d}/{NUM_EPISODES} | "
                f"Total Reward: {total_reward:8.2f} | "
                f"Actor Loss: {actor_loss:6.4f} | "
                f"Critic Loss: {critic_loss:6.4f}"
            )

        # Save model checkpoint periodically
        if episode % 50 == 0:
            ckpt_path = CHECKPOINT_DIR / f"mappo_ep{episode}.pt"
            policy.save(ckpt_path)
            logger.info(f"Saved model checkpoint to '{ckpt_path}'")

    logger.info("MAPPO training run completed successfully.")


if __name__ == "__main__":
    main()