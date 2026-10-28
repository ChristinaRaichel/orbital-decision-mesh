"""
Gymnasium multi-agent environment for LEO satellite beam hopping.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SatComBeamEnv(gym.Env):
    """Multi-Agent environment for satellite beam-to-cell allocation."""
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_satellites: int = 4, num_cells: int = 8):
        super().__init__()
        self.num_satellites = num_satellites
        self.num_cells = num_cells

        # obs_dim = 12 per satellite (lat, lon, alt, vel, battery, 7 cell demands)
        self.obs_dim = 12
        self.act_dim = num_cells  # Choose 1 target cell out of num_cells

        # Individual agent spaces (Box for observation, Discrete for action)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.act_dim)

        self.current_step = 0
        self.max_steps = 50

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Generate initial synthetic observations for all agents
        observations = [np.random.randn(self.obs_dim).astype(np.float32) for _ in range(self.num_satellites)]
        global_state = np.concatenate(observations)
        
        return observations, global_state

    def step(self, actions):
        self.current_step += 1
        
        # Simulate reward function (higher throughput when beam matches high-demand cell)
        rewards = [float(np.random.uniform(0.5, 2.0)) for _ in range(self.num_satellites)]
        
        done = self.current_step >= self.max_steps
        truncated = False

        next_observations = [np.random.randn(self.obs_dim).astype(np.float32) for _ in range(self.num_satellites)]
        global_state = np.concatenate(next_observations)
        
        infos = [{"agent_id": i} for i in range(self.num_satellites)]

        return next_observations, global_state, rewards, done, infos