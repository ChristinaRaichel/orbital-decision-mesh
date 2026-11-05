"""
src/observability/metrics.py
Prometheus metrics emitters for real-time constellation telemetry, policy optimization, and MCP latency monitoring.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Tier 1: Control Plane Metrics (MAPPO Policy)
MAPPO_EPISODE_REWARD = Gauge("mappo_episode_reward_total", "Total cumulative episodic reward for MAPPO policy")
MAPPO_ACTOR_LOSS = Gauge("mappo_actor_loss", "Policy surrogate loss for MAPPO actor")
MAPPO_CRITIC_LOSS = Gauge("mappo_critic_loss", "Value function mean squared error for MAPPO critic")
MAPPO_ACTION_ENTROPY = Gauge("mappo_action_entropy", "Action distribution entropy for exploration monitoring")

# Tier 2: Data Quality & Drift Metrics (Evidently AI)
EVIDENTLY_DATA_DRIFT_SCORE = Gauge("evidently_data_drift_score", "Share of drifted satellite OMM features [0.0 - 1.0]")
EVIDENTLY_DRIFT_ALERT_COUNT = Counter("evidently_drift_alerts_total", "Total number of statistical drift alerts triggered")
SATELLITE_ALTITUDE_MEAN_KM = Gauge("constellation_satellite_mean_altitude_km", "Mean constellation orbital altitude in km")

# Tier 3: Agentic Gateway Metrics (FastMCP & Slow Loop)
MCP_TOOL_LATENCY_SECONDS = Histogram(
    "mcp_tool_execution_latency_seconds",
    "FastMCP tool execution duration in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)
RAG_RETRIEVAL_HIT_COUNT = Counter("rag_retrieval_hits_total", "Total successfully retrieved RAG telemetry context documents")
LLM_TOKEN_USAGE_COUNTER = Counter("copilot_llm_tokens_total", "Total LLM tokens consumed by Ground Telemetry Copilot", ["token_type"])

def start_metrics_exporter(port: int = 8000):
    """Starts the Prometheus metrics HTTP endpoint."""
    start_http_server(port)
    print(f"[Prometheus Exporter] Exposing metrics at http://localhost:{port}/metrics")