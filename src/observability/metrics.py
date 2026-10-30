"""
Prometheus metric emitters for real-time training loop and system observability.
"""
from prometheus_client import Counter, Gauge, start_http_server

class TrainingMetrics:
    """Manager for emitting real-time telemetry and training metrics via Prometheus."""
    
    def __init__(self, port: int = 8000, start_server: bool = False):
        if start_server:
            try:
                start_http_server(port)
            except Exception:
                # Avoid binding errors if metrics server is already running
                pass
        
        # Prometheus Gauges and Counters
        self.episode_reward = Gauge("sat_episode_reward", "Total reward per training episode", ["agent_id"])
        self.actor_loss = Gauge("sat_actor_loss", "PPO Actor loss magnitude")
        self.critic_loss = Gauge("sat_critic_loss", "PPO Critic loss magnitude")
        self.drift_score = Gauge("sat_telemetry_drift_score", "Evidently AI dataset drift share")
        self.training_steps = Counter("sat_training_steps_total", "Total environment steps executed")

    def log_reward(self, agent_id: int, reward: float):
        """Record episode reward for a specific satellite agent."""
        self.episode_reward.labels(agent_id=str(agent_id)).set(reward)

    def log_losses(self, actor_loss: float, critic_loss: float):
        """Record optimization losses from the MAPPO update step."""
        self.actor_loss.set(actor_loss)
        self.critic_loss.set(critic_loss)

    def log_drift(self, score: float):
        """Record the current data drift score share from Evidently AI."""
        self.drift_score.set(score)

    def step(self):
        """Increment global step counter."""
        self.training_steps.inc()