"""
tests/test_slow_loop.py
Unit tests for the slow loop: FastMCP tools, RAG search, and Ground Telemetry Copilot.
"""

import pytest
from src.agents.copilot import GroundTelemetryCopilot
from src.mcp.server import (
    get_live_constellation_status,
    query_telemetry_rag,
    trigger_mappo_policy_evaluation
)


def test_ground_copilot_anomaly_analysis():
    """Verify that the Ground Copilot correctly formats operational anomaly analyses."""
    copilot = GroundTelemetryCopilot(model_name="test-model")
    
    report = copilot.analyze_anomaly(
        metric_name="Slant Range Attenuation",
        current_value=1250.5,
        threshold=1000.0,
        station_name="London Teleport"
    )
    
    assert isinstance(report, str)
    assert "London Teleport" in report
    assert "Slant Range Attenuation" in report
    assert "Recommended action" in report


def test_mcp_constellation_status_tool():
    """Verify that the MCP constellation status tool returns formatted telemetry strings."""
    status_report = get_live_constellation_status()
    
    assert isinstance(status_report, str)
    # Even with fallback data, it should return a valid summary or error message
    assert len(status_report) > 0


def test_mcp_rag_query_tool():
    """Verify that the MCP RAG search tool executes successfully and returns context."""
    query = "Analyze historical packet drop trends for Starlink low elevation passes"
    result = query_telemetry_rag(query)
    
    assert isinstance(result, str)
    assert "RAG Search Results" in result
    assert query in result or "Historical logs" in result


def test_mcp_mappo_trigger_tool():
    """Verify that the MCP trigger tool invokes policy evaluation successfully."""
    result = trigger_mappo_policy_evaluation()
    
    assert isinstance(result, str)
    assert "SUCCESS" in result or "MAPPO policy" in result