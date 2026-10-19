"""Read, interpolate, and export PHOENIX 2013 (ACES) and SYNMAST spectra."""

import itertools
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from astropy.io import fits
import h5py
import numpy as np
import pandas as pd
from ldtk.core import LOGG_POINTS, TEFF_POINTS, Z_POINTS
from ldtk.client import Client as LDTkClient

from .utilities.rds3v import rds3v
from .spectral_data import NativeSpectrum

logger = logging.getLogger(__name__)


def spectrum_filename(
    source,
    representation,
    temperature,
    logg,
    metallicity,
    band=None,
    wavelength_min=None,
    wavelength_max=None,
    velocity_step=None,
    vrt=None,
    alpha=None,
    magnetic_field=None,
):
    """Name spectra by their source, physical parameters, and preparation settings."""

    # Keep distinct sampling settings distinct in filenames.
    def number(value):
        return np.format_float_positional(float(value), unique=True, trim="-")

    parts = ["phx_aces" if source == "phx" else source]
    if band is not None:
        parts.append(str(band))
    parts.extend(
        [
            representation,
            f"T{number(temperature)}",
            f"logg{number(logg)}",
            f"metallicity{number(metallicity)}",
        ]
    )
    if alpha is not None:
        parts.append(f"alpha{number(alpha)}")
    if magnetic_field is not None:
        parts.append(f"B{number(magnetic_field)}G")
    if wavelength_min is not None:
        parts.append(f"w{number(wavelength_min)}-{number(wavelength_max)}")
    if velocity_step is not None:
        parts.append(f"dv{number(velocity_step)}")
    if vrt is not None:
        parts.append(f"vrt{number(vrt)}")
    return "_".join(parts) + ".h5"


def prepared_spectrum_path(
    star,
    directory,
    source,
    representation,
    temperature,
    band,
    wavelength_min,
    wavelength_max,
    velocity_step,
    magnetic_field=None,
):
    """Get the output path for a spectrum on the simulation velocity grid."""
    filename = spectrum_filename(
        source,
        representation,
        temperature,
        star.logg,
        star.metallicity,
        band=band,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        velocity_step=velocity_step,
        magnetic_field=magnetic_field,
        vrt=2.0 if source == "synmast" and representation == "flux" else None,
    )
    # Local spectra and CCF reference spectra have separate physical roles.
    cache_root = Path(star.out_path)
    if directory != "rv_ccf_templates":
        cache_root /= "local_spectra"
    path = cache_root / directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def require_spectral_path(paths: dict[str, Path], name: str) -> Path:
    """Get a path needed by a reader or tool, with a clear missing-path error."""
    if name not in paths:
        raise ValueError(
            f"Missing path {name!r}; set it in [paths] or the selected [paths:NAME]."
        )
    return paths[name]


def vac2air(wavelength):
    """Convert vacuum wavelengths to air wavelengths, both in Angstroms.

    lambda_air = lambda_vac / n; s = 1e4 / lambda_vac.
    VALD3 uses Morton (2000, ApJS, 130, 403), based on the refractive-index
    formula of Birch and Downs (1994, Metrologia, 31, 315).
    Source: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    """
    s = 1.0e4 / wavelength  # Inverse wavelength in micrometres^-1.
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength / n_air


def air2vac(wavelength):
    """Convert air wavelengths to vacuum wavelengths, both in Angstroms.

    lambda_vac = lambda_air * n; s = 1e4 / lambda_air.
    Uses the inverse solution derived by N. Piskunov for VALD3's
    Morton (2000) / Birch and Downs (1994) vacuum-to-air relation.
    Source: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    """
    s = 1.0e4 / wavelength  # Inverse wavelength in micrometres^-1.
    n_air = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    return wavelength * n_air


def phx2013_grid_filename(
    temp, logg_val, metallicity_val, alpha_val, spectrum_mode="flux"
):
    """Return the PHOENIX 2013 (ACES) subdirectory and original FITS filename."""
    if metallicity_val <= 0.0:
        prefix_subdir = "Z-{:.1f}".format(np.abs(metallicity_val))
        if alpha_val == 0.0:
            name = "lte0{}-{:.2f}-{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
                int(temp), logg_val, np.abs(metallicity_val)
            )
        elif alpha_val > 0.0:
            name = "lte0{}-{:.2f}-{:.1f}.Alpha=+{:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
                int(temp), logg_val, np.abs(metallicity_val), alpha_val
            )
        else:
            name = "lte0{}-{:.2f}-{:.1f}.Alpha={:.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
                int(temp), logg_val, np.abs(metallicity_val), alpha_val
            )
    else:
        prefix_subdir = "Z+{:.1f}".format(np.abs(metallicity_val))
        name = "lte0{}-{:.2f}+{:.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
            int(temp), logg_val, np.abs(metallicity_val)
        )

    if spectrum_mode == "int":
        name = name.replace("HiRes", "SpecInt")
    return prefix_subdir, name


PHX2013_FLUX_LOGG_GRID = np.linspace(0.0, 6.0, 13)
PHX2013_FLUX_METALLICITY_GRID = np.concatenate(
    (np.linspace(-4.0, -3.0, 2), np.linspace(-2.0, 1.0, 7))
)
PHX2013_FLUX_METALLICITY_GRID = PHX2013_FLUX_METALLICITY_GRID.round(decimals=2)
PHX2013_ALPHA_GRID = -0.2 + np.arange(8) * 0.2
PHX2013_ALPHA_GRID = PHX2013_ALPHA_GRID.round(decimals=2)

# PHOENIX 2013 spacing is 100 K below 7000 K and 200 K above it.
PHX2013_FLUX_TEMPERATURE_GRID = np.concatenate(
    [
        np.arange(2300.0, 7000.0, 100.0),
        np.arange(7000.0, 11800.0, 200.0),
    ]
)
# Keep the existing exclusion of the 5000 K model.
PHX2013_FLUX_TEMPERATURE_GRID = PHX2013_FLUX_TEMPERATURE_GRID[
    PHX2013_FLUX_TEMPERATURE_GRID != 5000.0
]

# Official intensity axes, checked on 2026-10-13:
# ftp://phoenix.astro.physik.uni-goettingen.de/v3.0/SpecInt50FITS/
PHX2013_OFFICIAL_INTENSITY_TEMPERATURE_GRID = np.concatenate([
    np.arange(2300.0, 7001.0, 100.0),
    np.arange(7200.0, 12001.0, 200.0),
    np.arange(12500.0, 15001.0, 500.0),
])
PHX2013_OFFICIAL_INTENSITY_LOGG_GRID = np.arange(0.0, 6.6, 0.5)
PHX2013_OFFICIAL_INTENSITY_METALLICITY_GRID = np.array(
    [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
)
# Share intensity axes with the installed LDTk, which fills the same source cache.
# https://github.com/hpparvi/ldtk/blob/master/ldtk/core.py
# Some combinations still lack files (e.g. 3700 K, logg=3.5, metallicity=0).
PHX2013_INTENSITY_TEMPERATURE_GRID = np.intersect1d(
    TEFF_POINTS, PHX2013_OFFICIAL_INTENSITY_TEMPERATURE_GRID
)
PHX2013_INTENSITY_LOGG_GRID = np.intersect1d(
    LOGG_POINTS, PHX2013_OFFICIAL_INTENSITY_LOGG_GRID
)
PHX2013_INTENSITY_METALLICITY_GRID = np.intersect1d(
    Z_POINTS, PHX2013_OFFICIAL_INTENSITY_METALLICITY_GRID
)

# --- Flux spectra (all mu integrated) ---
PHX2013_FLUX_BASE_URL = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"
PHX2013_WAVELENGTH_BASE_URL = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/"
PHX2013_WAVELENGTH_FILENAME = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"


def _download_if_missing(url, destination):
    """Download a missing source file without leaving a partial final file."""
    destination = Path(destination)
    if destination.is_file():
        logger.debug("Source file exists: %s", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    logger.info("Downloading PHOENIX source file: %s", destination.name)
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--show-error",
                url,
                "--output",
                str(temporary_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temporary_path.replace(destination)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or "").strip()
        raise RuntimeError(
            f"Failed to download {destination} from {url}: {details}"
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Could not download {destination} from {url}: {error}"
        ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _get_neighbors(value, grid):
    """Return bounding grid values and a linear weight, clamping at the edges."""
    if not np.isfinite(value):
        raise ValueError(f"Grid coordinate must be finite; got {value!r}.")
    if value <= grid[0]:
        return grid[0], grid[0], 0.0
    if value >= grid[-1]:
        return grid[-1], grid[-1], 0.0
    index = np.searchsorted(grid, value, side="right") - 1
    low, high = grid[index], grid[index + 1]
    return low, high, (value - low) / (high - low)


def _get_interpolation_corners(values, grids):
    """Yield surrounding grid points, weighting each by the product of the
    linear interpolation weights for temperature, logg, metallicity, and B
    when present.
    """
    bounds = [_get_neighbors(value, grid) for value, grid in zip(values, grids)]
    for coordinates in itertools.product(
        *(sorted({low, high}) for low, high, _ in bounds)
    ):
        weight = 1.0
        for coordinate, (low, high, fraction) in zip(coordinates, bounds):
            weight *= fraction if coordinate == high and low != high else 1.0 - fraction
        if weight >= 1e-5:
            yield tuple(float(coordinate) for coordinate in coordinates), float(weight)


def _interpolate_grid_spectra(corners, read_corner, *, require_matching_mu=True):
    """Sum corner spectra in float64, requiring matching mu unless opted out."""
    wavelength = mu_grid = values = None
    records = []
    for coordinates, weight in corners:
        if len(coordinates) == 4:
            logger.debug(
                "Interpolation corner: T=%g K, logg=%g, metallicity=%g, B=%g G; weight=%.8g",
                coordinates[0], coordinates[1], coordinates[3], coordinates[2], weight,
            )
        else:
            logger.debug(
                "Interpolation corner: T=%g K, logg=%g, metallicity=%g; weight=%.8g",
                *coordinates, weight,
            )
        wave, data, mu, path = read_corner(coordinates)
        data = np.asarray(data, dtype=np.float64)
        # Use the first spectrum's wavelength and viewing-angle grids as references.
        if wavelength is None:
            wavelength = np.asarray(wave).copy()
            mu_grid = None if mu is None else np.asarray(mu).copy()
            values = np.zeros_like(data, dtype=np.float64)
        # Require matching wavelength grids and array shapes; do not resample here.
        if np.shape(wave) != wavelength.shape or not np.allclose(
            wave, wavelength, rtol=1e-10, atol=0.0
        ):
            raise ValueError(
                "Interpolation-grid spectra have different wavelength grids."
            )
        if data.shape != values.shape:
            raise ValueError("Interpolation-grid spectra have different array shapes.")
        if (mu is None) != (mu_grid is None) or (
            mu is not None
            and (
                np.shape(mu) != mu_grid.shape
                or (
                    require_matching_mu
                    and not np.allclose(mu, mu_grid, rtol=1e-10, atol=0.0)
                )
            )
        ):
            raise ValueError("Interpolation-grid spectra have different mu grids.")
        # Apply one grid-point weight to the whole flux or intensity array.
        values += weight * data
        # Record the parameters, weight, and source file for each contribution.
        records.append(
            dict(coordinates=coordinates, weight=weight, file=str(Path(path).resolve()))
        )
    if wavelength is None:
        raise ValueError("No valid interpolation grid points were found.")
    return wavelength, values, mu_grid, records


def _grid_metadata(records, temperature, logg, metallicity, library_path):
    """Record interpolation inputs and the actual source files."""
    coordinates = np.asarray([record["coordinates"] for record in records])
    return {
        "grid_T_low": float(coordinates[:, 0].min()),
        "grid_T_high": float(coordinates[:, 0].max()),
        "grid_logg_low": float(coordinates[:, 1].min()),
        "grid_logg_high": float(coordinates[:, 1].max()),
        "grid_metallicity_low": float(coordinates[:, -1].min()),
        "grid_metallicity_high": float(coordinates[:, -1].max()),
        "target_Teff": float(temperature),
        "target_logg": float(logg),
        "target_metallicity": float(metallicity),
        "library_path": str(Path(library_path).expanduser().resolve()),
        "source_file_low": records[0]["file"],
        "source_file_high": records[-1]["file"],
        "interpolation_corners_json": json.dumps(records),
    }


def load_phx2013_flux_spectrum(temp, logg, metallicity, alpha, phx_path):
    """Read PHOENIX 2013 flux and interpolate temperature, logg, and metallicity."""
    logger.debug("PHOENIX 2013 flux target: T=%g K, logg=%g, metallicity=%g, alpha=%g",
                 temp, logg, metallicity, alpha)
    wavelength_min, wavelength_max = 3000, 30000
    flux_directory = Path(phx_path).expanduser() / "SpecFlux"

    wave_file = flux_directory / PHX2013_WAVELENGTH_FILENAME

    _download_if_missing(
        f"{PHX2013_WAVELENGTH_BASE_URL.rstrip('/')}/{PHX2013_WAVELENGTH_FILENAME}", wave_file
    )

    wave = fits.getdata(wave_file)
    selected = (wave >= wavelength_min - 500) & (wave <= wavelength_max + 500)
    grid_alpha = PHX2013_ALPHA_GRID[np.argmin(np.abs(PHX2013_ALPHA_GRID - alpha))]

    def read_corner(coordinates):
        temperature, gravity, abundance = coordinates
        selected_alpha = 0.0 if abundance > 0 else grid_alpha
        directory, filename = phx2013_grid_filename(
            temperature, gravity, abundance, selected_alpha
        )
        path = flux_directory / directory / filename
        _download_if_missing(
            f"{PHX2013_FLUX_BASE_URL.rstrip('/')}/{directory}/{filename}", path
        )
        logger.debug("Reading PHOENIX flux: %s", path.absolute())
        return wave[selected], fits.getdata(path)[selected], None, path

    corners = _get_interpolation_corners(
        (temp, logg, metallicity),
        (PHX2013_FLUX_TEMPERATURE_GRID, PHX2013_FLUX_LOGG_GRID,
         PHX2013_FLUX_METALLICITY_GRID),
    )
    wavelength, flux, _, records = _interpolate_grid_spectra(corners, read_corner)
    metadata = _grid_metadata(records, temp, logg, metallicity, phx_path)
    metadata.update(
        library="PHOENIX-ACES",
        target_alpha=float(alpha),
        source_wave_file=str(wave_file.resolve()),
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )
    return NativeSpectrum(
        wavelength=wavelength,
        values=flux,
        representation="flux",
        source="phx",
        metadata=metadata,
    )


def _reuse_legacy_phx2013_continuum(metadata, out_path, filename):
    """Copy a matching old continuum fit into the current cache layout."""
    _, name = phx2013_grid_filename(
        metadata["target_Teff"],
        metadata["target_logg"],
        metadata["target_metallicity"],
        metadata["target_alpha"],
    )
    name = name.replace(".fits", "").replace("/", "")
    data_name = f"wv{metadata['wavelength_min']}_{metadata['wavelength_max']}_{name}.h5"
    # Preserve existing continuum fits when changing the output layout.
    legacy = Path(out_path).parent / "syn_spec" / "RASSINE_PHX" / f"RASSINE_{data_name}"
    continuum = (
        Path(out_path) / "local_spectra" / "rassine_continuum" / filename.replace(".h5", "_rassine.h5")
    )
    same_grid_parameters = all(
        metadata.get(f"grid_{name}_{side}", target) == target
        for name, target in (
            ("logg", metadata["target_logg"]),
            ("metallicity", metadata["target_metallicity"]),
        )
        for side in ("low", "high")
    )
    if same_grid_parameters and legacy.is_file() and not continuum.exists():
        continuum.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, continuum)


def save_phx2013_flux_spectrum(spectrum, out_path):
    """Export PHOENIX 2013 flux as the wave/flux table used by RASSINE.

    Returns the HDF path. This export is optional for simulation
    spectra but is required by the current RASSINE template pipeline.
    Table columns and numerical precision are retained.
    """
    if spectrum.source != "phx" or spectrum.representation != "flux":
        raise ValueError("Expected a native PHOENIX flux spectrum.")
    metadata = spectrum.metadata
    filename = spectrum_filename(
        "phx",
        "flux",
        metadata["target_Teff"],
        metadata["target_logg"],
        metadata["target_metallicity"],
        alpha=metadata["target_alpha"],
        wavelength_min=metadata["wavelength_min"],
        wavelength_max=metadata["wavelength_max"],
    )
    out_name = Path(out_path) / "local_spectra" / "library_interpolated" / filename
    _reuse_legacy_phx2013_continuum(metadata, out_path, filename)
    out_name.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "wave": spectrum.wavelength,
            "flux": spectrum.values,
        }
    ).to_hdf(out_name, key="spec", complevel=5, index=False)
    return str(out_name)


def _resolve_phx2013_intensity_corner(coordinates, alpha, library_path, client):
    """Download an official corner, replacing only its temperature if absent."""
    temperature, gravity, abundance = coordinates
    directory, filename = phx2013_grid_filename(
        temperature, gravity, abundance, alpha, spectrum_mode="int"
    )
    server_files = set(client.files_in_server.get(directory, ()))
    selected_temperature = temperature
    if filename not in server_files:
        # Keep gravity and metallicity fixed. Equal-distance ties use the cooler model.
        candidates = sorted(PHX2013_INTENSITY_TEMPERATURE_GRID,
                            key=lambda value: (abs(value - temperature), value))
        for candidate in candidates:
            _, candidate_filename = phx2013_grid_filename(
                candidate, gravity, abundance, alpha, spectrum_mode="int"
            )
            if candidate_filename in server_files:
                selected_temperature, filename = float(candidate), candidate_filename
                break
        else:
            raise FileNotFoundError(
                f"No official PHOENIX intensity model at logg={gravity}, "
                f"metallicity={abundance}, alpha={alpha}; cannot replace temperature."
            )
        logger.warning(
            "Missing official PHOENIX intensity corner: T=%g K, logg=%g, "
            "metallicity=%g; using T=%g K at the same logg and metallicity. "
            "Check results near this grid gap.",
            temperature, gravity, abundance, selected_temperature,
        )
    path = library_path / directory / filename
    if not path.is_file():
        # Use LDTk's downloader and source cache; local absence is not a grid hole.
        client.set_limits((selected_temperature, selected_temperature),
                          (gravity, gravity), (abundance, abundance))
        client.download_uncached_files()
    if not path.is_file():
        raise FileNotFoundError(f"PHOENIX intensity file unavailable after download: {path}")
    return path, selected_temperature


def load_phx2013_intensity_spectrum(temp, logg, metallicity, alpha, library_path=None):
    """Interpolate PHOENIX 2013 intensity in temperature, logg, and metallicity."""
    logger.debug("PHOENIX 2013 intensity target: T=%g K, logg=%g, metallicity=%g, alpha=%g",
                 temp, logg, metallicity, alpha)
    library_path = Path(
        "~/.ldtk/cache_visir-lowres" if library_path is None else library_path
    ).expanduser()
    grid_alpha = PHX2013_ALPHA_GRID[np.argmin(np.abs(PHX2013_ALPHA_GRID - alpha))]
    client = None
    replaced_corners = []

    def read_corner(coordinates):
        nonlocal client
        temperature, gravity, abundance = coordinates
        selected_alpha = 0.0 if abundance > 0 else grid_alpha
        directory, filename = phx2013_grid_filename(
            temperature, gravity, abundance, selected_alpha, spectrum_mode="int"
        )
        path = library_path / directory / filename
        if not path.is_file():
            if client is None:
                library_path.mkdir(parents=True, exist_ok=True)
                client = LDTkClient(dataset="visir-lowres", cache=str(library_path), verbosity=0)
            path, selected_temperature = _resolve_phx2013_intensity_corner(
                coordinates, selected_alpha, library_path, client
            )
            if selected_temperature != temperature:
                replaced_corners.append({
                    "requested_temperature": temperature,
                    "used_temperature": selected_temperature,
                    "logg": gravity, "metallicity": abundance, "source_file": str(path),
                })
        logger.debug("Reading PHOENIX intensity: %s", path.absolute())
        with fits.open(path) as hdul:
            header = hdul[0].header
            wave = header["crval1"] + np.arange(header["naxis1"]) * header["cdelt1"]
            # Retain the existing nine center-to-limb samples.
            mu = hdul[1].data[-17::2]
            data = hdul[0].data[-17::2, :]
            selected = (wave >= 0) & (wave <= 50500)
            return wave[selected], data[:, selected].T, mu, path

    corners = _get_interpolation_corners(
        (temp, logg, metallicity),
        (PHX2013_INTENSITY_TEMPERATURE_GRID, PHX2013_INTENSITY_LOGG_GRID,
         PHX2013_INTENSITY_METALLICITY_GRID),
    )
    # PHOENIX mu values vary between models. Preserve the existing interpolation
    # by column position, using the first (lowest-temperature) corner's mu.
    wave, intensity, mu, records = _interpolate_grid_spectra(
        corners, read_corner, require_matching_mu=False
    )
    metadata = _grid_metadata(records, temp, logg, metallicity, library_path)
    metadata.update(library="PHOENIX-ACES", target_alpha=float(alpha))
    if replaced_corners:
        # Store structured records as JSON, like interpolation_corners_json.
        metadata["temperature_replacements_json"] = json.dumps(replaced_corners)
    return NativeSpectrum(
        wavelength=wave,
        values=intensity,
        mu=mu,
        representation="intensity",
        source="phx",
        metadata=metadata,
    )


def _save_intensity_hdf(spectrum, output_path, *, transpose=False, shuffle=False):
    """Write wavelength, mu, intensity, and metadata in the requested layout."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = spectrum.values.T if transpose else spectrum.values
    with h5py.File(output_path, "w") as handle:
        # Keep the existing dataset name "flux" for both intensity file layouts.
        for key, data in (
            ("wave", spectrum.wavelength),
            ("mu", spectrum.mu),
            ("flux", values),
        ):
            handle.create_dataset(
                key, data=data, compression="gzip", compression_opts=5, shuffle=shuffle
            )
        for key, value in spectrum.metadata.items():
            handle.attrs[key] = value
    return str(output_path)


def save_phx2013_intensity_spectrum(spectrum, out_path):
    """Export PHOENIX 2013 intensity with the existing (n_mu, n_wave) layout.

    Used for optional debug output; simulation reads NativeSpectrum directly.
    """
    if (
        spectrum.source != "phx"
        or spectrum.representation != "intensity"
        or spectrum.mu is None
    ):
        raise ValueError("Expected a native PHOENIX intensity spectrum with mu.")
    metadata = spectrum.metadata
    filename = spectrum_filename(
        "phx",
        "intensity",
        metadata["target_Teff"],
        metadata["target_logg"],
        metadata["target_metallicity"],
        alpha=metadata["target_alpha"],
    )
    out_name = Path(out_path) / "local_spectra" / "library_interpolated" / filename
    return _save_intensity_hdf(spectrum, out_name, transpose=True)


def parse_synmast_filename(name: str):
    """Read temperature, logg, metallicity, and B in Gauss from a grid filename."""
    # Accept the optional a+ suffix before _b or after the field value.
    match = re.match(
        r"^mc_(\d{4})(\d{3})([+-]\d{3})(?:a\+)?_b(\d+)(?:a\+)?_.*\.prf\.gz$",
        Path(name).name,
    )
    if match is None:
        raise ValueError(f"Unexpected SYNMAST .prf.gz filename: {name}")
    temperature, gravity, metallicity, magnetic_field = match.groups()
    return (
        int(temperature),
        int(gravity) / 100.0,
        int(metallicity) / 100.0,
        int(magnetic_field) * 100.0,
    )


def _synmast_grid_files(library_path):
    """Index actual filenames by temperature, logg, B, and metallicity."""
    files = {}
    for path in sorted(Path(library_path).glob("*.prf.gz")):
        try:
            temperature, gravity, abundance, field = parse_synmast_filename(path.name)
        except ValueError:
            continue
        key = (temperature, gravity, field, abundance)
        if key in files:
            raise ValueError(
                f"Duplicate SYNMAST grid point {key}: {files[key]} and {path}"
            )
        files[key] = path
    if not files:
        raise FileNotFoundError(
            f"No recognized SYNMAST .prf.gz files found in {library_path}"
        )
    return files


def _load_synmast_spectrum(
    temp, logg, metallicity, library_path, magnetic_field, representation
):
    """Interpolate SYNMAST spectra using the actual four-dimensional grid files."""
    logger.debug("SYNMAST %s target: T=%g K, logg=%g, metallicity=%g, B=%g G",
                 representation, temp, logg, metallicity, magnetic_field)
    library_path = Path(library_path).expanduser()
    files = _synmast_grid_files(library_path)
    grids = [np.array(sorted({key[axis] for key in files})) for axis in range(4)]
    corners = list(
        _get_interpolation_corners((temp, logg, magnetic_field, metallicity), grids)
    )
    # Require all contributing corners; never silently substitute a different model.
    for coordinates, _ in corners:
        if coordinates not in files:
            raise FileNotFoundError(
                f"Missing SYNMAST interpolation corner {coordinates} in {library_path}"
            )

    def read_corner(coordinates):
        path = files[coordinates]
        logger.debug("Reading SYNMAST %s: %s", representation, path.absolute())
        if representation == "intensity":
            # rds3v returns before vrt broadening in intensity mode; still pending.
            wave, mu, values, _ = rds3v(str(path), vsini=0, vrt=2.0, get_intensity=True)
            values = values[:, None] if values.ndim == 1 else values
        else:
            wave, values, _ = rds3v(str(path), vsini=0, vrt=2.0, get_intensity=False)
            mu = None
            if values.ndim != 1:
                raise ValueError("SYNMAST flux spectra must be one-dimensional arrays.")
        return wave, values, mu, path

    wave, values, mu, records = _interpolate_grid_spectra(corners, read_corner)
    metadata = _grid_metadata(records, temp, logg, metallicity, library_path)
    metadata.update(
        target_magnetic_field_G=float(magnetic_field),
        grid_magnetic_field_low_G=min(record["coordinates"][2] for record in records),
        grid_magnetic_field_high_G=max(record["coordinates"][2] for record in records),
    )
    return NativeSpectrum(
        wavelength=air2vac(wave).astype(np.float32, copy=False),
        values=values,
        representation=representation,
        source="synmast",
        mu=None if mu is None else np.asarray(mu, dtype=np.float32),
        values_unit="erg s^-1 cm^-2 cm^-1" if representation == "flux" else None,
        metadata=metadata,
    )


def load_synmast_intensity_spectrum(
    temp, logg, metallicity, library_path, magnetic_field=0.0
):
    """Interpolate unnormalized SYNMAST intensity in T, logg, metallicity, and B."""
    return _load_synmast_spectrum(
        temp, logg, metallicity, library_path, magnetic_field, "intensity"
    )


def save_synmast_intensity_spectrum(spectrum, out_path, band_name=None):
    """Export SYNMAST intensity with the existing (n_wave, n_mu) layout.

    Used for optional debug output; filenames include the band and B.
    """
    if (
        spectrum.source != "synmast"
        or spectrum.representation != "intensity"
        or spectrum.mu is None
    ):
        raise ValueError("Expected a native SYNMAST intensity spectrum with mu.")
    metadata = spectrum.metadata
    filename = spectrum_filename(
        "synmast",
        "intensity",
        metadata["target_Teff"],
        metadata["target_logg"],
        metadata["target_metallicity"],
        band=band_name,
        magnetic_field=metadata.get("target_magnetic_field_G", 0.0),
    )
    out_name = Path(out_path) / "local_spectra" / "library_interpolated" / filename
    return _save_intensity_hdf(spectrum, out_name, shuffle=True)


def load_synmast_flux_spectrum(
    temp, logg, metallicity, library_path, magnetic_field=0.0
):
    """Interpolate unnormalized SYNMAST flux in T, logg, metallicity, and B.

    rds3v integrates each corner with zero rotation and vrt=2 km/s.
    """
    return _load_synmast_spectrum(
        temp, logg, metallicity, library_path, magnetic_field, "flux"
    )
