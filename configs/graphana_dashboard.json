"""
src/observability/drift.py
Evidently AI data drift detection pipeline comparing baseline training OMM telemetry with runtime batches.
"""

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.observability.metrics import (
    EVIDENTLY_DATA_DRIFT_SCORE,
    EVIDENTLY_DRIFT_ALERT_COUNT,
    SATELLITE_ALTITUDE_MEAN_KM
)


def run_drift_analysis(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_dir: str = "docs/assets/reports") -> dict:
    """
    Executes DataDriftPreset on satellite telemetry, updates Prometheus metrics, 
    and generates an HTML report artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate drift metrics
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    report_dict = drift_report.as_dict()
    drift_share = report_dict["metrics"][0]["result"]["share_of_drifted_columns"]
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    
    # Update Prometheus gauges
    EVIDENTLY_DATA_DRIFT_SCORE.set(drift_share)
    if "sat_alt_km" in current_df.columns:
        SATELLITE_ALTITUDE_MEAN_KM.set(current_df["sat_alt_km"].mean())
        
    if dataset_drift:
        EVIDENTLY_DRIFT_ALERT_COUNT.inc()
    
    # Save standalone HTML dashboard artifact
    report_path = os.path.join(output_dir, "evidently_drift_report.html")
    drift_report.save_html(report_path)
    
    return {
        "dataset_drift_detected": dataset_drift,
        "drift_share": drift_share,
        "report_path": report_path
    }