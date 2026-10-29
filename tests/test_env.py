"""
Unit tests for the multi-agent satellite Gymnasium environment.
"""

import pytest
import numpy as np
from src.envs.sat_env import SatComBeamEnv

def test_env_initialization():
    env = SatComBeamEnv(num_satellites=4, num_cells=8)
    assert env.num_satellites == 4
    assert env.num_cells == 8
    assert env.obs_dim == 12
    assert env.act_dim == 8

def test_env_reset():
    env = SatComBeamEnv(num_satellites=4, num_cells=8)
    observations, global_state = env.reset()
    
    assert isinstance(observations, list)
    assert len(observations) == 4
    for obs in observations:
        assert obs.shape == (12,)
        assert obs.dtype == np.float32
    
    assert isinstance(global_state, np.ndarray)
    assert global_state.shape == (48,)  # 4 satellites * 12 obs_dim

def test_env_step():
    env = SatComBeamEnv(num_satellites=4, num_cells=8)
    env.reset()
    actions = [env.action_space.sample() for _ in range(4)]
    next_obs, global_state, rewards, done, infos = env.step(actions)
    
    assert len(next_obs) == 4
    assert len(rewards) == 4
    assert isinstance(done, bool)
    assert len(infos) == 4