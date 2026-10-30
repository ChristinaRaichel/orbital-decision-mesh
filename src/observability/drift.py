"""
Evidently AI integration for monitoring data drift in satellite streaming telemetry.
"""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

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
        result = self.report.as_dict()
        
        # Extract top-level dataset drift metrics
        drift_result = result["metrics"][0]["result"]
        drift_summary = {
            "dataset_drift": drift_result.get("dataset_drift", False),
            "share_of_drifted_columns": drift_result.get("share_of_drifted_columns", 0.0),
            "number_of_columns": drift_result.get("number_of_columns", 0),
            "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0)
        }
        return drift_summary