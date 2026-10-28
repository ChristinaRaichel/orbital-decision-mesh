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


def compute_gae(
    rewards,
    values,
    next_value,
    dones,
    gamma=0.99,
    lam=0.95,
):
    """
    rewards:    [T, N]
    values:     [T, N], [T, 1], or [T]
    next_value: scalar, [1], or [N]
    dones:      [T, N], [T, 1], or [T]
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    device = rewards.device
    T, N = rewards.shape

    values = torch.as_tensor(
        values, dtype=torch.float32, device=device
    )

    # Normalize values to [T, N].
    if values.dim() == 1:
        values = values.unsqueeze(-1)

    values = values.expand(T, N)

    next_value = torch.as_tensor(
        next_value, dtype=torch.float32, device=device
    ).reshape(-1)

    if next_value.numel() == 1:
        next_value = next_value.expand(N)
    elif next_value.numel() != N:
        raise ValueError(
            f"next_value must contain 1 or {N} elements, "
            f"got {next_value.numel()}"
        )

    dones = torch.as_tensor(
        dones, dtype=torch.float32, device=device
    )

    if dones.dim() == 1:
        dones = dones.unsqueeze(-1)

    dones = dones.expand(T, N)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, dtype=torch.float32, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        non_terminal = 1.0 - dones[t]

        delta = (
            rewards[t]
            + gamma * next_val * non_terminal
            - values[t]
        )

        last_gae = (
            delta
            + gamma * lam * non_terminal * last_gae
        )

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
    # Safe extraction that handles both Box and Tuple/List spaces without crashing
    obs_dim = env.observation_space[0].shape[0] if hasattr(env.observation_space, '__getitem__') else env.observation_space.shape[-1]
    act_dim = env.action_space[0].n if hasattr(env.action_space, '__getitem__') else env.action_space.n
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
            next_global_tensor = torch.as_tensor(
                global_state,
                dtype=torch.float32,
                device=device,
            )
            next_value = policy.get_value(next_global_tensor)

        # Stack collected rollout data.
        # Observations:    [T, N, obs_dim]
        # Actions:         [T, N]
        # Log probabilities: [T, N]
        # Rewards:         [T, N]
        # Values:          [T, ...]
        # Dones:           [T, ...]

        ep_obs_tensor = torch.stack(ep_obs)
        ep_actions_tensor = torch.stack(ep_actions)
        ep_log_probs_tensor = torch.stack(ep_log_probs)
        ep_rewards_tensor = torch.stack(ep_rewards)
        ep_values_tensor = torch.stack(ep_values)
        ep_dones_tensor = torch.stack(ep_dones)
        ep_global_states_tensor = torch.stack(ep_global_states)

        T = ep_rewards_tensor.shape[0]
        N = num_sats

        # Normalize rewards to [T, N].
        ep_rewards_tensor = ep_rewards_tensor.reshape(T, N)

        # Calculate GAE BEFORE flattening the rollout.
        advantages, returns = compute_gae(
            rewards=ep_rewards_tensor,
            values=ep_values_tensor,
            next_value=next_value,
            dones=ep_dones_tensor,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )

        # Flatten time and agent dimensions: T * N samples.
        b_obs = ep_obs_tensor.reshape(T * N, obs_dim)
        b_actions = ep_actions_tensor.reshape(T * N)
        b_old_log_probs = ep_log_probs_tensor.reshape(T * N)

        b_returns = returns.reshape(T * N)
        b_advantages = advantages.reshape(T * N)

        # Repeat each centralized global state for every agent.
        # [T, global_state_dim] -> [T, N, global_state_dim]
        b_global_states = (
            ep_global_states_tensor
            .unsqueeze(1)
            .expand(T, N, global_state_dim)
            .reshape(T * N, global_state_dim)
        )

        # Ensure all PPO inputs have matching batch sizes.
        assert b_obs.shape[0] == T * N
        assert b_actions.shape[0] == T * N
        assert b_old_log_probs.shape[0] == T * N
        assert b_global_states.shape[0] == T * N
        assert b_returns.shape[0] == T * N
        assert b_advantages.shape[0] == T * N

        # Update MAPPO policy.
        actor_loss, critic_loss = policy.update(
            b_obs,
            b_global_states,
            b_actions,
            b_old_log_probs,
            b_returns,
            b_advantages,
            epochs=PPO_EPOCHS,
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