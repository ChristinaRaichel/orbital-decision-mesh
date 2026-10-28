"""
Multi-Agent Proximal Policy Optimization (MAPPO) actor-critic policy architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


class MAPPOPolicy:
    def __init__(self, obs_dim, act_dim, global_state_dim, lr=3e-4, device="cpu"):
        self.device = device
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(global_state_dim).to(device)
        
        self.optimizer = optim.Adam([
            {"params": self.actor.parameters(), "lr": lr},
            {"params": self.critic.parameters(), "lr": lr * 2}
        ])
        self.clip_eps = 0.2
        self.entropy_coef = 0.01

    def get_actions(self, obs_tensor, global_state_tensor):
        dist = self.actor(obs_tensor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(global_state_tensor)
        return actions, log_probs, values

    def get_value(self, global_state_tensor):
        return self.critic(global_state_tensor)


    def update(
    self,
    b_obs,
    b_global_states,
    b_actions,
    b_old_log_probs,
    b_returns,
    b_advantages,
    epochs=4,
):
        # Inputs are already flattened by train.py.
        # b_obs:           [batch_size, obs_dim]
        # b_global_states: [batch_size, global_state_dim]
        # Other tensors:   [batch_size]

        obs = b_obs.reshape(-1, b_obs.shape[-1])
        g_states = b_global_states.reshape(
            -1, b_global_states.shape[-1]
        )
        actions = b_actions.reshape(-1).long()
        old_log_probs = b_old_log_probs.reshape(-1)
        returns = b_returns.reshape(-1)
        advantages = b_advantages.reshape(-1)

        batch_size = obs.shape[0]

        # Verify all PPO inputs have the same batch size.
        tensors = {
            "global_states": g_states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
        }

        for name, tensor in tensors.items():
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: obs={batch_size}, "
                    f"{name}={tensor.shape[0]}"
                )

        # Normalize advantages.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(epochs):
            # Actor update.
            dist = self.actor(obs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                )
                * advantages
            )

            actor_loss = (
                -torch.min(surr1, surr2).mean()
                - self.entropy_coef * entropy
            )

            # Centralized critic update.
            values = self.critic(g_states)
            critic_loss = nn.functional.mse_loss(
                values, returns
            )

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return (
            total_actor_loss / epochs,
            total_critic_loss / epochs
        )

    
    def save(self, path):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict()
        }, path)