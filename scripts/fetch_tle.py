"""
scripts/fetch_tle.py
Standalone utility to fetch live Starlink orbital parameters, compute sample ground station
visibility, and save an offline JSON snapshot.
"""

import sys
import json
import logging
from pathlib import Path

# Ensure 'src' is in Python path when executing from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import SatelliteDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_tle")

OUTPUT_CACHE_PATH = Path("data/starlink_cache.json")

# Ground Stations for coverage verification (London, New York, Tokyo)
TEST_GROUND_STATIONS = [
    {"name": "London Teleport", "lat": 51.5074, "lon": -0.1278},
    {"name": "New York Gateway", "lat": 40.7128, "lon": -74.0060},
    {"name": "Tokyo Station", "lat": 35.6762, "lon": 139.6503},
]


def main():
    logger.info("Starting satellite telemetry ingestion run...")
    
    # 1. Initialize Loader
    loader = SatelliteDataLoader(limit_sats=5)
    satellites = loader.fetch_active_starlink_sats()

    if not satellites:
        logger.error("No satellite records retrieved. Check Space-Track credentials or network.")
        sys.exit(1)

    logger.info(f"Loaded {len(satellites)} operational satellite objects.")

    # 2. Compute Visibility Across Target Ground Stations
    print("\n" + "=" * 65)
    print("           REAL-TIME SATELLITE VISIBILITY REPORT           ")
    print("=" * 65)

    cached_telemetry = []
    for sat in satellites:
        sat_data = {"sat_name": sat.name, "stations": {}}
        print(f"\nSatellite: {sat.name}")
        
        for station in TEST_GROUND_STATIONS:
            vis = loader.compute_ground_visibility(
                sat=sat,
                ground_lat=station["lat"],
                ground_lon=station["lon"],
                min_elevation_deg=10.0
            )
            sat_data["stations"][station["name"]] = vis
            
            status = "IN RANGE" if vis["is_visible"] else "OUT OF RANGE"
            print(
                f"  -> {station['name']:<18} | Az: {vis['azimuth_deg']:>6.1f}° | "
                f"El: {vis['elevation_deg']:>5.1f}° | Dist: {vis['distance_km']:>7.1f} km | [{status}]"
            )
            
        cached_telemetry.append(sat_data)

    print("\n" + "=" * 65)

    # 3. Export Offline Cache
    OUTPUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CACHE_PATH, "w") as f:
        json.dump(cached_telemetry, f, indent=2)

    logger.info(f"Saved local telemetry cache to '{OUTPUT_CACHE_PATH}'.")


if __name__ == "__main__":
    main()