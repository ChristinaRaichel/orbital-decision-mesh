"""
Prometheus and LangSmith telemetry observability hooks.
"""

import logging
from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger("observability")

# Prometheus Metrics
BEAM_HOP_REWARD = Gauge('beam_hop_episode_reward', 'Total reward per episode')
DATA_DRIFT_SCORE = Gauge('evidently_data_drift_score', 'Feature drift score detected by Evidently')

def start_metrics_server(port=9090):
    """Starts background Prometheus metrics emitter server."""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.warning(f"Could not start metrics server: {e}")