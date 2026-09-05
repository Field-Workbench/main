"""Geomagnetic helpers used by Background field scene objects.

Field Workbench stores the resolved background vector in scene coordinates so
saved scenes remain reproducible. WMM2025 is evaluated only when the user asks
for it (creation/recalculation), via the small ``pygeomag`` dependency and the
official WMM2025 coefficient set it distributes.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

WMM2025_START_DATE = date(2025, 1, 1)
WMM2025_END_DATE = date(2029, 12, 31)
WMM2025_MODEL_NAME = "WMM-2025"


def decimal_year(value: date) -> float:
    """Return a calendar date as the fractional year expected by WMM."""
    start = date(value.year, 1, 1)
    end = date(value.year + 1, 1, 1)
    return value.year + (value - start).days / float((end - start).days)


def _validate_latlon(latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    latitude = float(latitude_deg)
    longitude = float(longitude_deg)
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("Latitude must be between -90 and +90 degrees.")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("Longitude must be between -180 and +180 degrees.")
    return latitude, longitude


def latlon_to_utm(latitude_deg: float, longitude_deg: float) -> dict[str, Any]:
    """Convert WGS84 latitude/longitude to UTM without an external GIS package."""
    latitude, longitude = _validate_latlon(latitude_deg, longitude_deg)
    if not -80.0 <= latitude <= 84.0:
        raise ValueError("UTM is defined here only between 80°S and 84°N.")

    zone = int((longitude + 180.0) / 6.0) + 1
    zone = min(60, max(1, zone))
    # Standard UTM special zones for southwest Norway and Svalbard.
    if 56.0 <= latitude < 64.0 and 3.0 <= longitude < 12.0:
        zone = 32
    if 72.0 <= latitude < 84.0:
        if 0.0 <= longitude < 9.0:
            zone = 31
        elif 9.0 <= longitude < 21.0:
            zone = 33
        elif 21.0 <= longitude < 33.0:
            zone = 35
        elif 33.0 <= longitude < 42.0:
            zone = 37

    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996

    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    lon_origin = (zone - 1) * 6 - 180 + 3
    lon_origin_rad = math.radians(lon_origin)

    n = a / math.sqrt(1.0 - ecc_sq * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = ecc_prime_sq * math.cos(lat_rad) ** 2
    aa = math.cos(lat_rad) * (lon_rad - lon_origin_rad)

    m = a * (
        (1 - ecc_sq / 4 - 3 * ecc_sq**2 / 64 - 5 * ecc_sq**3 / 256) * lat_rad
        - (3 * ecc_sq / 8 + 3 * ecc_sq**2 / 32 + 45 * ecc_sq**3 / 1024)
        * math.sin(2 * lat_rad)
        + (15 * ecc_sq**2 / 256 + 45 * ecc_sq**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * ecc_sq**3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ecc_prime_sq) * aa**5 / 120
    ) + 500000.0
    northing = k0 * (
        m
        + n
        * math.tan(lat_rad)
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ecc_prime_sq) * aa**6 / 720
        )
    )
    hemisphere = "N" if latitude >= 0.0 else "S"
    if latitude < 0.0:
        northing += 10000000.0
    return {
        "zone": zone,
        "hemisphere": hemisphere,
        "easting_m": float(easting),
        "northing_m": float(northing),
    }


def utm_to_latlon(
    zone: int,
    easting_m: float,
    northing_m: float,
    hemisphere: str = "N",
) -> tuple[float, float]:
    """Convert a WGS84 UTM coordinate to latitude/longitude."""
    zone_number = int(zone)
    if not 1 <= zone_number <= 60:
        raise ValueError("UTM zone must be between 1 and 60.")
    easting = float(easting_m)
    northing = float(northing_m)
    if not math.isfinite(easting) or not 0.0 <= easting <= 1_000_000.0:
        raise ValueError("UTM easting must be between 0 and 1,000,000 m.")
    if not math.isfinite(northing) or not 0.0 <= northing <= 10_000_000.0:
        raise ValueError("UTM northing must be between 0 and 10,000,000 m.")
    hemi = str(hemisphere).strip().upper()
    if hemi not in {"N", "S"}:
        raise ValueError("UTM hemisphere must be N or S.")

    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996

    x = easting - 500000.0
    y = northing - (10_000_000.0 if hemi == "S" else 0.0)
    lon_origin = (zone_number - 1) * 6 - 180 + 3

    m = y / k0
    mu = m / (a * (1 - ecc_sq / 4 - 3 * ecc_sq**2 / 64 - 5 * ecc_sq**3 / 256))
    e1 = (1 - math.sqrt(1 - ecc_sq)) / (1 + math.sqrt(1 - ecc_sq))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    n1 = a / math.sqrt(1 - ecc_sq * sin_fp**2)
    r1 = a * (1 - ecc_sq) / (1 - ecc_sq * sin_fp**2) ** 1.5
    t1 = tan_fp**2
    c1 = ecc_prime_sq * cos_fp**2
    d = x / (n1 * k0)

    latitude = fp - (n1 * tan_fp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ecc_prime_sq) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ecc_prime_sq - 3 * c1**2)
        * d**6
        / 720
    )
    longitude = math.radians(lon_origin) + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ecc_prime_sq + 24 * t1**2)
        * d**5
        / 120
    ) / cos_fp

    lat_deg = math.degrees(latitude)
    lon_deg = math.degrees(longitude)
    _validate_latlon(lat_deg, lon_deg)
    return float(lat_deg), float(lon_deg)


def scene_vector_from_ned(
    north_uT: float,
    east_uT: float,
    down_uT: float,
    *,
    heading_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Transform a local N/E/Down field into Workbench X/Y/Z coordinates.

    At zero orientation Workbench +X is east, +Y is true north, and +Z is up.
    ``heading`` is clockwise from true north. The optional pitch is then applied
    about the resulting scene +X axis and roll about the resulting scene +Y
    axis. The returned components are in the same units as the inputs.
    """
    world_enu = np.asarray([east_uT, north_uT, -down_uT], dtype=float)
    if not np.isfinite(world_enu).all():
        raise ValueError("Geomagnetic vector components must be finite.")

    h = math.radians(float(heading_deg))
    p = math.radians(float(pitch_deg))
    r = math.radians(float(roll_deg))
    if not all(math.isfinite(value) for value in (h, p, r)):
        raise ValueError("Heading, pitch, and roll must be finite.")

    # scene -> local ENU. Heading is clockwise, hence the negative conventional
    # Z rotation. Pitch/roll are intrinsic scene-axis rotations in this order.
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rz = np.asarray([[ch, sh, 0.0], [-sh, ch, 0.0], [0.0, 0.0, 1.0]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.asarray([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    scene_to_enu = rz @ rx @ ry
    return scene_to_enu.T @ world_enu


def wmm2025_field(
    *,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float = 0.0,
    when: date | None = None,
    heading_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> dict[str, Any]:
    """Evaluate WMM2025 and return both geographic and Workbench vectors."""
    latitude, longitude = _validate_latlon(latitude_deg, longitude_deg)
    altitude = float(altitude_m)
    if not math.isfinite(altitude):
        raise ValueError("Altitude must be finite.")
    # WMM itself supports a much larger height range; these generous bounds
    # catch unit mistakes while still covering terrestrial and balloon work.
    if not -1000.0 <= altitude <= 850_000.0:
        raise ValueError("Altitude must be between -1,000 m and 850,000 m.")
    sample_date = when or date.today()
    if not WMM2025_START_DATE <= sample_date <= WMM2025_END_DATE:
        raise ValueError("WMM2025 dates must be between 2025-01-01 and 2029-12-31.")

    try:
        from pygeomag import GeoMag
        from pygeomag.wmm.wmm_2025 import WMM_2025
    except ImportError as error:  # pragma: no cover - packaging/dependency failure
        raise RuntimeError(
            "WMM2025 support requires pygeomag. Install the Field Workbench requirements."
        ) from error

    value = decimal_year(sample_date)
    result = GeoMag(coefficients_data=WMM_2025).calculate(
        glat=latitude,
        glon=longitude,
        alt=altitude / 1000.0,
        time=value,
    )
    north_uT = float(result.x) / 1000.0
    east_uT = float(result.y) / 1000.0
    down_uT = float(result.z) / 1000.0
    scene = scene_vector_from_ned(
        north_uT,
        east_uT,
        down_uT,
        heading_deg=heading_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    return {
        "model": WMM2025_MODEL_NAME,
        "date": sample_date.isoformat(),
        "decimal_year": value,
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "altitude_m": altitude,
        "north_uT": north_uT,
        "east_uT": east_uT,
        "down_uT": down_uT,
        "horizontal_uT": float(result.h) / 1000.0,
        "total_uT": float(result.f) / 1000.0,
        "declination_deg": float(result.d),
        "inclination_deg": float(result.i),
        "heading_deg": float(heading_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "scene_vector_uT": scene.astype(float).tolist(),
    }
