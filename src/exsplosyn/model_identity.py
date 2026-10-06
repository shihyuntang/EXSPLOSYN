"""Create a shared physical model ID for pairing simulation output files.

Hash the effective stellar and active-region parameters, including region order
and disk grid size. Combine a readable prefix such as T3600i70 with the first
10 hash characters. CSV and spectrum files use this ID to identify outputs
from the same physical model within an output directory.

The ID does not depend on config formatting, phases, spectral bands or output
switches. Repeating the same physical model reuses its ID, allowing matching
output files to be overwritten. model_ids.json records the full hashes and
rejects a short-ID collision before the corresponding outputs are written.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _numbers(value):
    """
    Use the same JSON numbers for integers/floats and
    positive/negative zero.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        number = float(array)
        return 0.0 if number == 0 else number
    return [_numbers(item) for item in array]


def assign_model_identity(star, active_region_box, grid_size):
    """
    Hash effective physical inputs in original region order,
    plus disk grid size.
    """
    longitude, latitude, radius, temperature = active_region_box

    physical_temperatures = getattr(star, "region_temperatures", None)
    parameters = {
        "photosphere_temperature_k": _numbers(star.photosphere_temperature),
        "stellar_radius_km": _numbers(getattr(star, "rad", None)),
        "rotation_period_days": _numbers(star.prot),
        "inclination_angle_deg": _numbers(star.inclination_angle),
        "logg": _numbers(star.logg),
        "metallicity": _numbers(star.metallicity),
        "photosphere_B_gauss": _numbers(getattr(star, "photosphere_B", 0)),
        "active_B_gauss": _numbers(getattr(star, "active_B", 0)),
        "region_longitude_deg": _numbers(longitude),
        "region_latitude_deg": _numbers(latitude),
        "region_radius_fraction": _numbers(radius),
        "region_temperature_k": _numbers(
            physical_temperatures if physical_temperatures is not None else temperature
        ),
        "region_black_mask": np.asarray(
            getattr(star, "region_black_mask", np.zeros(len(longitude), dtype=bool)),
            dtype=bool,
        ).tolist(),
        "grid_size": int(grid_size),
    }
    # Gaussian regions use brightness instead of physical temperatures.
    brightness = getattr(star, "gaussian_brightness", {})
    if brightness:
        parameters["gaussian_region_brightness"] = {
            band: _numbers(value) for band, value in sorted(brightness.items())
        }
    # Fixed key order and spacing make the hash independent of config formatting.
    serialized_parameters = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    full_hash = hashlib.sha256(serialized_parameters.encode("utf-8")).hexdigest()
    star.model_hash = full_hash
    readable_prefix = (
        f"T{parameters['photosphere_temperature_k']:g}"
        f"i{parameters['inclination_angle_deg']:g}"
    )
    star.model_id = f"{readable_prefix}_{full_hash[:10]}"
    return star.model_id


def register_model_identity(directory, star):
    """Check full hashes before allowing same-ID model outputs to be overwritten."""
    # Store full hashes so two models with the same short ID cannot overwrite each other.
    registry_path = Path(directory) / "model_ids.json"
    identities = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists()
        else {}
    )
    previous = identities.get(star.model_id)
    if previous is not None and previous != star.model_hash:
        raise ValueError(
            f"Short model ID collision: {star.model_id}. Existing outputs were not overwritten."
        )
    if previous is None:
        identities[star.model_id] = star.model_hash
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(identities, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def identity_arrays(star):
    """Store identifiers as scalar strings readable without pickle."""
    if not hasattr(star, "model_id"):
        return {}
    return {
        "model_id": np.asarray(star.model_id),
        "model_hash": np.asarray(star.model_hash),
    }


def identity_suffix(star):
    """Return the filename suffix _id-<model_id>, or empty if no ID is assigned."""
    return f"_id-{star.model_id}" if hasattr(star, "model_id") else ""
