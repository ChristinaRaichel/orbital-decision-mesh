"""
FastMCP server exposing telemetry ingestion and beam allocation controls to LLM ground copilot.
"""

import logging
from fastmcp import FastMCP
from src.data.loader import SatelliteDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("Constellation Beam-Hopping Server")
loader = SatelliteDataLoader(limit_sats=5)

@mcp.tool()
def get_live_constellation_status() -> str:
    """Fetch active Starlink satellite telemetry and check London visibility."""
    satellites = loader.fetch_active_starlink_sats()
    if not satellites:
        return "Error: Could not retrieve satellite records from Space-Track."
    
    report = []
    for sat in satellites:
        vis = loader.compute_ground_visibility(sat, ground_lat=51.5074, ground_lon=-0.1278)
        report.append(
            f"Sat: {vis['sat_name']} | Alt: {vis['sat_alt_km']} km | "
            f"El: {vis['elevation_deg']}° | Visible: {vis['is_visible']}"
        )
    return "\n".join(report)


if __name__ == "__main__":
    mcp.run()