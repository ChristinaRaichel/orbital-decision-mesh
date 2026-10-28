"""
Data ingestion module for satellite orbital parameters via Space-Track API.
Uses EarthSatellite.from_omm to prevent TLE string column-offset corruptions.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
from skyfield.api import load, EarthSatellite, wgs84

load_dotenv()
logger = logging.getLogger(__name__)

SPACE_TRACK_URL = "https://www.space-track.org"

class SatelliteDataLoader:
    """Handles fetching and parsing satellite orbital telemetry."""

    def __init__(self, limit_sats: int = 10):
        self.limit_sats = limit_sats
        self.ts = load.timescale()
        self.username = os.getenv("SPACETRACK_USERNAME")
        self.password = os.getenv("SPACETRACK_PASSWORD")

    def fetch_active_starlink_sats(self) -> List[EarthSatellite]:
        """Queries Space-Track for active, non-decayed Starlink satellites."""
        if not self.username or not self.password:
            logger.warning("Space-Track credentials missing. Loading fallback offline satellites.")
            return self._generate_fallback_satellites()

        session = requests.Session()
        login_url = f"{SPACE_TRACK_URL}/ajaxauth/login"
        credentials = {"identity": self.username, "password": self.password}

        # Query active operational Starlinks (PERIAPSIS > 300km, DECAY_DATE is null)
        query_url = (
            f"{SPACE_TRACK_URL}/basicspacedata/query/class/gp/"
            f"OBJECT_NAME/STARLINK~~/DECAY_DATE/null-val/PERIAPSIS/>300/"
            f"format/json/orderby/NORAD_CAT_ID/limit/{self.limit_sats}"
        )

        try:
            resp = session.post(login_url, data=credentials, timeout=10)
            if resp.status_code != 200:
                logger.error("Failed to authenticate with Space-Track.")
                return self._generate_fallback_satellites()

            data_resp = session.get(query_url, timeout=15)
            data_resp.raise_for_status()
            records = data_resp.json()

            satellites = []
            for item in records:
                try:
                    # Native OMM parsing bypasses raw TLE column bugs & Alpha-5 IDs
                    sat = EarthSatellite.from_omm(self.ts, item)
                    satellites.append(sat)
                except Exception as err:
                    logger.debug(f"Skipping record parsing: {err}")
                    continue

            logger.info(f"Loaded {len(satellites)} operational satellites from Space-Track.")
            return satellites

        except Exception as e:
            logger.error(f"Space-Track fetch error: {e}. Falling back to baseline data.")
            return self._generate_fallback_satellites()

    def compute_ground_visibility(
        self, sat: EarthSatellite, ground_lat: float, ground_lon: float, min_elevation_deg: float = 10.0
    ) -> Dict[str, Any]:
        """Calculates Azimuth, Elevation, and Line-of-Sight visibility for a Ground Station."""
        now = self.ts.now()
        ground_target = wgs84.latlon(ground_lat, ground_lon)
        
        # Calculate vector difference between satellite and ground station
        difference = sat - ground_target
        topocentric = difference.at(now)
        alt, az, distance = topocentric.altaz()

        # Extract sub-satellite point (directly overhead on ground)
        subpoint = wgs84.subpoint(sat.at(now))

        return {
            "sat_name": sat.name,
            "sat_lat": round(float(subpoint.latitude.degrees), 4),
            "sat_lon": round(float(subpoint.longitude.degrees), 4),
            "sat_alt_km": round(float(subpoint.elevation.km), 2),
            "azimuth_deg": round(float(az.degrees), 2),
            "elevation_deg": round(float(alt.degrees), 2),
            "distance_km": round(float(distance.km), 2),
            "is_visible": bool(alt.degrees >= min_elevation_deg),
        }

    def _generate_fallback_satellites(self) -> List[EarthSatellite]:
        """Synthetic offline baseline for testing environment without network dependency."""
        fallback_omm = [
            {
                "OBJECT_NAME": "STARLINK-SIM-1",
                "EPOCH": "2026-09-22T00:00:00.000000",
                "MEAN_MOTION": "15.06",
                "ECCENTRICITY": "0.0001",
                "INCLINATION": "53.05",
                "RA_OF_ASC_NODE": "120.1",
                "ARG_OF_PERICENTER": "80.0",
                "MEAN_ANOMALY": "280.0",
                "NORAD_CAT_ID": "99901",
                "BSTAR": "0.0001",
            }
        ]
        return [EarthSatellite.from_omm(self.ts, item) for item in fallback_omm]