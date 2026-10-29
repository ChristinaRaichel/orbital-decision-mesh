"""
Unit tests for the MAPPO Actor-Critic policy architecture and dimension bounds.
"""

import pytest
import torch
from src.models.mappo_policy import MAPPOPolicy

def test_mappo_policy_dimensions():
    obs_dim = 12
    act_dim = 8
    global_state_dim = 48
    device = "cpu"

    policy = MAPPOPolicy(obs_dim, act_dim, global_state_dim, device=device)

    batch_size = 5
    obs_tensor = torch.randn(batch_size, obs_dim, device=device)
    global_tensor = torch.randn(batch_size, global_state_dim, device=device)

    actions, log_probs, values = policy.get_actions(obs_tensor, global_tensor)

    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)

def test_critic_value_forward():
    global_state_dim = 48
    policy = MAPPOPolicy(obs_dim=12, act_dim=8, global_state_dim=global_state_dim)
    
    g_state = torch.randn(10, global_state_dim)
    val = policy.get_value(g_state)
    assert val.shape == (10,)