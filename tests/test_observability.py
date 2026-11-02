"""
Unit tests for Prometheus metrics emitters and Evidently AI drift monitoring.
"""

import pandas as pd
import pytest
from src.observability.metrics import TrainingMetrics
from src.observability.drift import TelemetryDriftMonitor

def test_training_metrics_initialization():
    """Verify Prometheus metrics manager initializes without error."""
    metrics = TrainingMetrics(port=8001, start_server=False)
    assert metrics is not None
    
    # Test metric logging functions
    metrics.log_reward(agent_id=0, reward=15.5)
    metrics.log_losses(actor_loss=0.45, critic_loss=1.2)
    metrics.log_drift(score=0.1)
    metrics.step()

def test_telemetry_drift_monitor():
    """Verify Evidently AI data drift monitor runs and returns correct metrics format."""
    # Create mock reference baseline telemetry data
    ref_data = pd.DataFrame({
        "altitude": [550.0, 551.2, 549.8, 550.5],
        "inclination": [53.0, 53.0, 53.1, 53.0],
        "cell_demand": [100.0, 105.0, 98.0, 102.0]
    })
    
    monitor = TelemetryDriftMonitor(reference_data=ref_data)
    
    # Create mock current streaming telemetry data (similar distribution)
    curr_data = pd.DataFrame({
        "altitude": [550.2, 550.8, 550.1, 550.3],
        "inclination": [53.0, 53.0, 53.0, 53.1],
        "cell_demand": [102.0, 104.0, 99.0, 101.0]
    })
    
    drift_summary = monitor.evaluate_drift(curr_data)
    
    assert isinstance(drift_summary, dict)
    assert "dataset_drift" in drift_summary
    assert "share_of_drifted_columns" in drift_summary
    assert drift_summary["number_of_columns"] == 3