"""
Evidently AI integration for monitoring data drift in satellite streaming telemetry.
"""
import json
import pandas as pd

# Fallback imports to handle version discrepancies in evidently package layouts
try:
    from evidently.report import Report
except ImportError:
    from evidently import Report

try:
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.presets import DataDriftPreset
    except ImportError:
        from evidently.metric_preset import DataDriftPreset  # type: ignore

class TelemetryDriftMonitor:
    """Monitors shifts in satellite orbital states and traffic cell demand distributions."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initializes the drift monitor with baseline reference telemetry data 
        (e.g., historical OMM parameters and normal traffic matrices).
        """
        self.reference_data = reference_data
        self.report = Report(metrics=[DataDriftPreset()])

    def evaluate_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Compares current streaming telemetry batch against reference data
        and extracts high-level drift summary metrics.
        """
        self.report.run(reference_data=self.reference_data, current_data=current_data)
        
        # Use report.json() parsed as dict to remain compatible across all Evidently v4+ releases
        try:
            result = json.loads(self.report.json())
        except Exception:
            result = self.report.as_dict() if hasattr(self.report, "as_dict") else {}
        
        # Extract top-level dataset drift metrics safely
        metrics_list = result.get("metrics", [])
        if not metrics_list:
            return {
                "dataset_drift": False,
                "share_of_drifted_columns": 0.0,
                "number_of_columns": len(current_data.columns),
                "number_of_drifted_columns": 0
            }
            
        drift_result = metrics_list[0].get("result", {})
        drift_summary = {
            "dataset_drift": drift_result.get("dataset_drift", False),
            "share_of_drifted_columns": drift_result.get("share_of_drifted_columns", 0.0),
            "number_of_columns": drift_result.get("number_of_columns", len(current_data.columns)),
            "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0)
        }
        return drift_summary