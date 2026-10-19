"""Read configuration settings and prepare simulation inputs."""

import configparser
import logging
import os
import re
from pathlib import Path

import numpy as np

from . import ccf_templates, spectral_io
from . import surface_integration as stsp
from .configuration import (
    MIN_TEMPERATURE_K,
    MAX_TEMPERATURE_K,
    resolve_config_file,
    resolve_temperature_table_file,
    resolve_spectral_paths,
    resolve_path_profile,
)
from .gaussian_profiles import read_gaussian_parameters, read_gaussian_brightness
from .star import Star
from .output_options import resolve_output_options
from .model_identity import assign_model_identity, register_model_identity

SOLAR_RADIUS = 695700  # km


def build_spectroscopic_bands(
    cfg: configparser.ConfigParser, paths: dict[str, Path] | None = None
):
    """Build spectroscopic bands from configuration.

    Wavelengths and window half-widths are expressed in Angstroms.
    Band identifiers and configuration order are preserved for spectral
    lookup and output naming.

    Returns:
        tuple: Band definitions, diagnostics, and SYNMAST library paths keyed by band identifier.
    """
    # --- Spectroscopic parameters ---
    # Read the enabled spectroscopic configurations.
    enabled_raw = cfg.get("spectroscopic", "enabled").strip()
    enabled_modes = [name.strip() for name in enabled_raw.split(",") if name.strip()]

    if len(enabled_modes) != len(set(enabled_modes)):
        raise ValueError("Each spectroscopic configuration may be enabled only once.")

    for name in enabled_modes:
        if name != "phx_rv" and not cfg.has_section(f"band:{name}"):
            raise ValueError(f"Missing configuration section [band:{name}].")

    # Use the PHOENIX grid parameters to build band definitions.
    # Wavelengths and window half-widths are expressed in Angstroms.
    if "phx_rv" in enabled_modes:
        wv_start = cfg.getint("phx_rv", "wavelength_start")
        wv_end = cfg.getint("phx_rv", "wavelength_end")
        wv_step = cfg.getint("phx_rv", "wavelength_step")
        sim_spec_window_size = cfg.getfloat("phx_rv", "window_half_width")

        if wv_step <= 0 or wv_end < wv_start:
            raise ValueError(
                "The PHOENIX RV wavelength grid requires a positive step "
                "and an end wavelength greater than or equal to the start."
            )
        if sim_spec_window_size <= 0:
            raise ValueError("The PHOENIX RV window half-width must be positive.")

        sim_spec_line_centers = np.arange(
            wv_start, wv_end + 1, wv_step, dtype=np.float32
        )

    # --- Bands setup ---
    # Build band definitions in the requested configuration order.
    # Each definition contains:
    # [center_A, half_width_A, mode, spectrum_type, spectrum_source].
    bands = {}

    band_diagnostics = {}
    band_library_paths = {}

    for mode_name in enabled_modes:
        if mode_name == "phx_rv":
            for center in sim_spec_line_centers:
                band_name = f"w{int(center) // 100}"

                if band_name in bands:
                    raise ValueError(
                        f"Duplicate band identifier {band_name!r}. "
                        "Check the wavelength spacing and enabled configurations."
                    )

                bands[band_name] = [
                    float(center),
                    sim_spec_window_size,
                    "rv",
                    "flux",
                    "phx",
                ]
                band_diagnostics[band_name] = ("rv",)
        else:
            section = f"band:{mode_name}"
            source = cfg.get(section, "source").strip().lower()

            if source == "gaussian_profile":
                parameters = read_gaussian_parameters(cfg, section)
                if mode_name in bands:
                    raise ValueError(f"Duplicate band identifier {mode_name!r}.")
                if cfg.has_option(section, "diagnostics"):
                    raise ValueError(
                        f"[{section}] Gaussian profiles use first-moment RV only; "
                        "omit diagnostics."
                    )
                bands[mode_name] = [
                    parameters.center,
                    parameters.wavelength_half_width,
                    "rv",
                    "flux",
                    source,
                ]
                band_diagnostics[mode_name] = ("rv",)
                continue

            if source != "synmast":
                raise ValueError(
                    f"[{section}] requires source = synmast or gaussian_profile. "
                    "Use [phx_rv] for the PHOENIX RV wavelength grid."
                )

            # The band references a key in the merged [paths] settings.
            path_key = cfg.get(section, "library_path", fallback="").strip().lower()
            if not path_key:
                raise ValueError(f"[{section}] requires library_path.")
            if paths is None:
                paths = resolve_spectral_paths(cfg, "default")
            if path_key not in paths:
                raise ValueError(
                    f"[{section}] library_path references missing path {path_key!r}; "
                    "set it in [paths] or the selected [paths:NAME]."
                )
            library_path = paths[path_key]

            center = cfg.getfloat(section, "center")
            half_width = cfg.getfloat(section, "window_half_width")

            if not np.isfinite(center) or center <= 0:
                raise ValueError(f"[{section}] center must be finite and positive.")
            if not np.isfinite(half_width) or half_width <= 0:
                raise ValueError(
                    f"[{section}] window_half_width must be finite " "and positive."
                )
            if center - half_width <= 0:
                raise ValueError(f"[{section}] wavelength window must be positive.")

            band_name = mode_name
            if band_name in bands:
                raise ValueError(f"Duplicate band identifier {band_name!r}.")

            spectrum_mode = cfg.get(section, "spectrum_mode", fallback="intensity").strip().lower()
            if spectrum_mode not in {"flux", "intensity"}:
                raise ValueError(f"[{section}] spectrum_mode must be flux or intensity.")
            bands[band_name] = [
                center,
                half_width,
                "rv",
                spectrum_mode,
                source,
            ]

            diagnostics_raw = cfg.get(section, "diagnostics", fallback="rv")
            diagnostics = tuple(
                name.strip() for name in diagnostics_raw.split(",") if name.strip()
            )

            allowed_diagnostics = {"rv"}
            if mode_name == "FeOH":
                allowed_diagnostics.add("ewr_FeOH")

            if not diagnostics:
                raise ValueError(f"[{section}] must select at least one diagnostic.")
            if len(diagnostics) != len(set(diagnostics)):
                raise ValueError(f"[{section}] contains duplicate diagnostics.")

            unsupported = set(diagnostics) - allowed_diagnostics
            if unsupported:
                raise ValueError(
                    f"Unsupported diagnostics for [{section}]: "
                    f"{', '.join(sorted(unsupported))}."
                )

            band_diagnostics[band_name] = diagnostics
            band_library_paths[band_name] = library_path

    return bands, band_diagnostics, band_library_paths


def resolve_table_temperatures(labels, photosphere_t, table_file):
    """Convert labels to region temperatures using the supplied offset table.

    The table contains signed temperature offsets, grouped by photospheric
    temperature. Labels are case-sensitive. No interpolation or nearest-row
    selection is performed.
    """
    table_file = Path(table_file).expanduser()
    table = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
    )
    # Preserve uppercase and lowercase labels.
    table.optionxform = str

    with table_file.open(encoding="utf-8") as stream:
        table.read_file(stream)

    temperature_section = str(int(photosphere_t))
    if not table.has_section(temperature_section):
        raise ValueError(
            f"No temperature table for {photosphere_t:g} K in {table_file}."
        )

    temperatures = []
    for label in labels:
        if not table.has_option(temperature_section, label):
            raise ValueError(
                f"Unknown temperature label {label!r} in section "
                f"[{temperature_section}] of {table_file}."
            )

        temperature_offset = table.getfloat(temperature_section, label)
        if not np.isfinite(temperature_offset):
            raise ValueError(f"Temperature offset for label {label!r} must be finite.")

        temperatures.append(photosphere_t + temperature_offset)

    return np.asarray(temperatures, dtype=np.float64)


def _setup_photometry(cfg, bands, logger):
    """Load cached filters and add the shared photometry band."""
    # --- Photometry parameters ---
    filters_raw = cfg.get("photometry", "filters", fallback="").strip()
    filters = [name.strip() for name in filters_raw.split(",")] if filters_raw else []
    has_filters = len(filters) > 0 and filters[0].lower() not in ("", "none")
    photometry_filters = {}

    if has_filters:
        logger.info("%27s: %s", "Photometry filters", ", ".join(filters))
        try:
            from pyphot import Filter, config, svo

            cache_dir = "./bandpass_cache"
            os.makedirs(cache_dir, exist_ok=True)

            wmin, wmax = np.inf, -np.inf

            for filter_name in filters:
                cache_filename = filter_name.replace("/", "_") + ".csv"
                cache_path = os.path.join(cache_dir, cache_filename)

                if not os.path.exists(cache_path):
                    # Download and cache the full filter profile.
                    logger.info("Downloading filter %s from SVO to cache...", filter_name)
                    filter_obj = svo.get_pyphot_filter(filter_name)

                    # Cache wavelengths in Angstroms.
                    wavelength_angstrom = filter_obj.wavelength.to("AA").value
                    filter_dtype = filter_obj.dtype  # 'energy' or 'photon'

                    save_data = np.column_stack(
                        (wavelength_angstrom, filter_obj.transmit)
                    )

                    # Record photon or energy weighting for loading the cached filter.
                    header_text = f"dtype:{filter_dtype}\nwavelength_AA,transmission"

                    np.savetxt(
                        cache_path,
                        save_data,
                        delimiter=",",
                        header=header_text,
                    )

                # Build filters from cached data using the photometry convention.
                with open(cache_path, encoding="utf-8") as cache_stream:
                    filter_dtype = cache_stream.readline().strip().split(":")[1]
                data = np.loadtxt(cache_path, delimiter=",")
                wavelength_angstrom = data[:, 0]

                photometry_filters[filter_name] = Filter(
                    wavelength_angstrom * config.units.U("AA"),
                    data[:, 1],
                    name=filter_name,
                    dtype=filter_dtype,
                )

                # Cover all selected filters using the existing float32 precision.
                wavelengths = np.asarray(wavelength_angstrom, dtype=np.float32)
                wmin = float(np.minimum(wmin, np.min(wavelengths)))
                wmax = float(np.maximum(wmax, np.max(wavelengths)))

            if wmin < 500 or wmax > 50000:
                logger.warning(
                    f"Filter wavelength range {wmin:.1f}–{wmax:.1f} Å is outside 500–50000 Å."
                )

            photo_center = int((wmax + wmin) * 0.5)
            photo_width = int((wmax - wmin) * 0.5)
            bands["photo"] = [
                float(photo_center),
                float(photo_width),
                "photo",
                "intensity",
                "phx",
            ]

        except Exception as e:
            logger.warning(
                f"Photometry filters could not be loaded ({e}). Photometry will be skipped."
            )
            has_filters = False
    else:
        logger.info("No filters specified. Will not generate photometry.")

    return filters, photometry_filters


def _calculate_velocity_step(instrument_res, spec_oversample_factor, logger):
    """Set spectral velocity spacing from the instrumental FWHM."""
    if spec_oversample_factor <= 0:
        raise ValueError("[advanced] spec_oversample_factor must be positive.")
    # --- Spectral velocity step ---
    c_kms = 299792.458
    if instrument_res > 0:
        inst_fwhm_kms = c_kms / instrument_res
        spectrum_vel_step = inst_fwhm_kms / spec_oversample_factor

        # Limit the maximum step for CCF sampling.
        if spectrum_vel_step > 1.0:
            spectrum_vel_step = 1.0

        # Limit memory use by keeping the step at least 0.1 km/s.
        elif spectrum_vel_step < 0.1:
            # Maximum velocity step for at least 3 samples per instrumental FWHM.
            three_point_step = inst_fwhm_kms / 3.0

            if 0.1 > three_point_step:
                logger.warning(
                    "At R=%d, the 0.1 km/s minimum gives fewer than 3 samples "
                    "per instrumental FWHM. Inspect spectral sampling; "
                    "the requested %d samples per FWHM cannot be reached.",
                    instrument_res,
                    spec_oversample_factor,
                )
            else:
                logger.info(
                    "Using the 0.1 km/s minimum at R=%d; the requested %d "
                    "samples per instrumental FWHM cannot be reached.",
                    instrument_res,
                    spec_oversample_factor,
                )
            spectrum_vel_step = 0.1
    else:
        # Default step when instrumental broadening is disabled.
        spectrum_vel_step = 1.0

    logger.debug("Spectral velocity step: %.3f km/s", spectrum_vel_step)

    return spectrum_vel_step


def _setup_star_and_regions(cfg, bands, config_file, logger):
    """Read stellar and region settings, retaining physical region temperatures."""
    # --- Stellar params ---
    stellar_radius = cfg.getfloat("star", "radius") * SOLAR_RADIUS
    rotation_period = cfg.getfloat("star", "prot")
    if not np.isfinite(stellar_radius) or stellar_radius <= 0:
        raise ValueError("Stellar radius must be finite and positive.")
    if not np.isfinite(rotation_period) or rotation_period <= 0:
        raise ValueError("Rotation period must be finite and positive.")
    inclination_angle = cfg.getfloat("star", "inclination_angle")

    photosphere_t = cfg.getfloat("star", "Tphot")
    if not np.isfinite(photosphere_t) or not MIN_TEMPERATURE_K <= photosphere_t <= MAX_TEMPERATURE_K:
        raise ValueError(
            f"Photospheric temperature must be between {MIN_TEMPERATURE_K} and {MAX_TEMPERATURE_K} K."
        )
    photosphere_t = float(np.round(photosphere_t))

    logg = cfg.getfloat("star", "logg")
    metallicity = cfg.getfloat("star", "metallicity")

    # Missing or blank magnetic-field settings mean zero Gauss.
    photosphere_B = float(cfg.get("advanced", "photosphere_B", fallback="").strip() or "0")
    active_B = float(cfg.get("advanced", "active_B", fallback="").strip() or "0")
    for name, value in (("photosphere_B", photosphere_B), ("active_B", active_B)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"[advanced] {name} must be finite and nonnegative (G).")

    def parse_region_list(section: str, key: str, dtype):
        """Read a comma- or whitespace-separated region list."""
        raw = cfg.get(section, key).strip()
        cleaned = re.sub(r"[,\s]+", " ", raw).strip()
        if not cleaned:
            return np.array([], dtype=dtype)

        return np.array(cleaned.split(), dtype=dtype)

    # --- Active regions ---
    section = "active_regions"
    # Gaussian-only runs use region brightness instead of temperature.
    needs_region_temperatures = any(
        params[4] != "gaussian_profile" for params in bands.values()
    )
    active_region_longitudes = parse_region_list(section, "longitudes", np.float32)
    active_region_latitudes = parse_region_list(section, "latitudes", np.float32)
    sizes_percent = parse_region_list(section, "sizes", np.float32)

    region_temperatures = None
    if needs_region_temperatures:
        temperature_mode = (
            cfg.get(section, "temperature_mode", fallback="absolute").strip().lower()
        )

        if temperature_mode not in {"absolute", "offset", "relative", "t_lookup_table"}:
            raise ValueError(
                f"Unsupported temperature_mode {temperature_mode!r}. "
                "Supported modes: absolute, offset, relative, t_lookup_table."
            )

        if temperature_mode == "absolute":
            region_temperatures = parse_region_list(section, "temperatures", np.float64)
        elif temperature_mode in {"offset", "relative"}:
            # Signed offsets are relative to the rounded photospheric temperature.
            temperature_offsets = parse_region_list(section, "temperature_offsets", np.float64)
            region_temperatures = photosphere_t + temperature_offsets
        else:
            region_labels = parse_region_list(section, "temperature_levels", str)
            table_raw = cfg.get(section, "temperature_table").strip()

            if not table_raw:
                raise ValueError("t_lookup_table mode requires a temperature_table path.")

            table_file = resolve_temperature_table_file(table_raw, config_file)

            region_temperatures = resolve_table_temperatures(
                region_labels,
                photosphere_t,
                table_file,
            )

    active_regions_number = active_region_longitudes.size

    if not (active_region_latitudes.size == sizes_percent.size == active_regions_number):
        raise ValueError(
            "Active-region longitude, latitude, and size lists "
            "must have equal lengths."
        )

    if not np.all(np.isfinite(sizes_percent)) or np.any(sizes_percent < 0):
        raise ValueError("Region sizes must be finite and nonnegative.")
    # Temporary limit until the large-region geometry issue is fixed.
    if np.any(sizes_percent > 80):
        raise ValueError(
            "Region sizes above 80% are currently unsupported due to a known "
            "geometry issue. Set each [active_regions] size to at most 80%."
        )
    # Internal non-emitting marker, above the allowed physical temperatures.
    BLACK_SPOT_SENTINEL = 99999
    black_threshold = None

    black_regions = np.zeros(active_regions_number, dtype=bool)
    cool_regions = black_regions.copy()
    hot_regions = black_regions.copy()
    neutral_regions = black_regions.copy()

    cool_temperatures = np.array([], dtype=np.float64)
    hot_temperatures = np.array([], dtype=np.float64)
    if needs_region_temperatures:
        if region_temperatures.size == 1 and active_regions_number > 1:
            region_temperatures = np.full(
                active_regions_number,
                region_temperatures[0],
                dtype=np.float64,
            )

        if region_temperatures.size != active_regions_number:
            raise ValueError(
                "Provide one temperature or offset for all regions, or "
                "one temperature or offset per region."
            )

        if not np.all(np.isfinite(region_temperatures)):
            raise ValueError("Region temperatures must be finite.")
        if np.any(region_temperatures <= 0):
            raise ValueError("Region temperatures must be positive.")
        if np.any(region_temperatures > MAX_TEMPERATURE_K):
            raise ValueError(f"Region temperatures must not exceed {MAX_TEMPERATURE_K} K.")

        region_temperatures = np.round(region_temperatures)

        if np.any(region_temperatures < 1):
            raise ValueError(
                "Region temperatures must remain at least 1 K after rounding."
            )

        # The threshold applies only to physical region temperatures.
        threshold_raw = cfg.get(
            "advanced", "black_below_temperature", fallback=""
        ).strip()

        if threshold_raw:
            threshold_value = float(threshold_raw)
            if not np.isfinite(threshold_value) or threshold_value < MIN_TEMPERATURE_K:
                raise ValueError(
                    f"black_below_temperature must be finite and at least {MIN_TEMPERATURE_K} K."
                )

            black_threshold = float(np.round(threshold_value))
            if black_threshold > photosphere_t:
                raise ValueError(
                    "black_below_temperature must not exceed "
                    "the photospheric temperature."
                )

            black_regions = region_temperatures < black_threshold

            logger.info(
                "Non-emitting threshold: %.0f K; selected %d region(s).",
                black_threshold,
                np.count_nonzero(black_regions),
            )

        physical_regions = ~black_regions
        if np.any(region_temperatures[physical_regions] < MIN_TEMPERATURE_K):
            raise ValueError(
                f"Emitting region temperatures must be at least {MIN_TEMPERATURE_K} K."
            )
        cool_regions = physical_regions & (region_temperatures < photosphere_t)
        hot_regions = physical_regions & (region_temperatures > photosphere_t)
        neutral_regions = physical_regions & (region_temperatures == photosphere_t)

        # Temperature alone selects the spectrum; equal temperatures must share B.
        uses_synmast = any(params[4].lower() == "synmast" for params in bands.values())
        if uses_synmast and photosphere_B != active_B and np.any(neutral_regions):
            raise ValueError(
                "SYNMAST regions at the photospheric temperature require "
                "active_B == photosphere_B. Temperatures are rounded to integer Kelvin."
            )

        cool_temperatures = region_temperatures[cool_regions]
        hot_temperatures = region_temperatures[hot_regions]

    # At disk center, projected area fraction = radius fraction squared.
    active_region_radii = np.sqrt(sizes_percent / 100.0, dtype=np.float32).astype(
        np.float64
    )

    active_region_Temp = (
        np.where(black_regions, BLACK_SPOT_SENTINEL, region_temperatures).astype(
            np.float32
        )
        if needs_region_temperatures
        else None
    )

    active_region_box = [
        active_region_longitudes,
        active_region_latitudes,
        active_region_radii,
        active_region_Temp,
    ]

    if needs_region_temperatures:
        logger.info(
            "%27s: %d cool, %d hot, %d neutral, %d black.",
            "Active regions",
            np.count_nonzero(cool_regions),
            np.count_nonzero(hot_regions),
            np.count_nonzero(neutral_regions),
            np.count_nonzero(black_regions),
        )

    else:
        logger.info(
            "%27s: %d; Gaussian brightness inputs only.",
            "Active regions",
            active_regions_number,
        )

    # --- Star object ---
    # Equatorial rotation speed; convert the period from days to seconds.
    rotation_vel = (2.0 * np.pi * stellar_radius) / (rotation_period * 86400.0)

    star = Star(
        prot=rotation_period,
        rotation_vel=rotation_vel,
        inclination_angle=inclination_angle,
        rad=stellar_radius,
        photosphere_temperature=photosphere_t,
        T_cool=cool_temperatures,
        T_hot=hot_temperatures,
        logg=logg,
        metallicity=metallicity,
        bands=bands,
    )
    # Project the equatorial rotation speed onto the line of sight.
    star.vsini = float(
        star.rotation_vel
        * np.sin(np.deg2rad(star.inclination_angle))
    )
    star.photosphere_B = photosphere_B
    star.active_B = active_B
    star.black_spot_sentinel = BLACK_SPOT_SENTINEL
    star.has_black_spots = bool(np.any(black_regions))

    # Keep actual region temperatures, including non-emitting regions.
    star.region_temperatures = (
        region_temperatures.copy()
        if needs_region_temperatures
        else None
    )

    # These arrays follow the original configuration order.
    star.region_black_mask = black_regions.copy()
    star.black_below_temperature = black_threshold

    return star, active_region_box


def _create_output_directories(save_dir, options):
    """Create output and spectral cache directories."""
    # --- Output and spectral cache directories ---
    output_directory = Path("outputs") / save_dir
    directories = [
        Path("spectral_cache/local_spectra/library_interpolated"),
        Path("spectral_cache/local_spectra/rassine_continuum"),
    ]
    if options.save_intermediate:
        directories.extend([
            Path("spectral_cache/local_spectra/simulation_ready"),
            Path("spectral_cache/rv_ccf_templates"),
        ])
    if not options.in_memory:
        directories.append(output_directory)
        if options.save_spectra:
            directories.extend([output_directory / "simulated_spotted_spectra", output_directory / "rv_ccf_templates"])
        if options.make_plots:
            directories.extend([output_directory / "continuum_fitting", Path("outputs/CCF")])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)



def _read_phases(cfg):
    """Read a phase file or use uniform phase sampling."""
    # --- Phase setup ---
    phase_file = cfg.get("phase_sampling", "phase_file", fallback="").strip()

    if phase_file and phase_file.lower() != "none":
        with open(phase_file, encoding="utf-8") as phase_stream:
            phases = np.fromiter(
                (float(line.split()[0]) for line in phase_stream), dtype=np.float32
            )
    else:
        phase_step = cfg.getfloat("phase_sampling", "phase_step")
        if not np.isfinite(phase_step) or phase_step <= 0:
            raise ValueError(
                "[phase_sampling] phase_step must be finite and positive when no phase_file is provided."
            )
        phases = np.arange(-0.75, 0.751, phase_step, dtype=np.float32)

    if phases.size == 0 or not np.all(np.isfinite(phases)):
        raise ValueError("Simulation phases must be nonempty and finite.")

    return phases


def _configure_rv_template(star, cfg, config_file):
    """Read RV template settings and return any requested PHOENIX temperature."""
    rv_template_temperature = None
    needs_ccf = any(
        params[4] != "gaussian_profile" and "rv" in star.band_diagnostics.get(band, ())
        for band, params in star.bands.items()
    )
    if needs_ccf:
        # Keep numerical sampling controls together in the advanced section.
        star.ccf_template_oversample_factor = cfg.getfloat(
            "advanced", "ccf_template_oversample_factor", fallback=5.0
        )
        star.ccf_rv_oversample_factor = cfg.getfloat(
            "advanced", "ccf_rv_oversample_factor", fallback=50.0
        )
        for name in ("ccf_template_oversample_factor", "ccf_rv_oversample_factor"):
            value = getattr(star, name)
            if not np.isfinite(value) or value < 1:
                raise ValueError(f"[advanced] {name} must be finite and at least 1.")
        # A fixed RV range sets the wavelength margin needed for template shifts.
        star.ccf_rv_half_range = cfg.getfloat("rv_ccf", "rv_half_range", fallback=40.0)
        if not np.isfinite(star.ccf_rv_half_range) or star.ccf_rv_half_range <= 1.0:
            raise ValueError(
                "[rv_ccf] rv_half_range must exceed the 1 km/s coarse step."
            )
        ccf_templates.configure_ccf_template(star, cfg, config_file)
        star.ccf_temp = star.photosphere_temperature
        if star.ccf_template_source == "phx":
            star.ccf_template_use_photosphere = cfg.getboolean(
                "rv_ccf", "template_use_photosphere", fallback=True
            )
            if not star.ccf_template_use_photosphere:
                template_temperature = cfg.getfloat("rv_ccf", "template_temperature")
                if (
                    not np.isfinite(template_temperature)
                    or template_temperature < MIN_TEMPERATURE_K
                    or template_temperature > MAX_TEMPERATURE_K
                ):
                    raise ValueError(
                        f"CCF template temperature must be between {MIN_TEMPERATURE_K} and {MAX_TEMPERATURE_K} K."
                    )

                star.ccf_temp = float(np.round(template_temperature))
                rv_template_temperature = star.ccf_temp

    return rv_template_temperature


def init_setup_cfg(
    logger: logging.Logger,
    config_file: str,
    output_suffix: str | None = None,
    *,
    path_profile: str | None = None,
    debug: bool = False,
    mode: str | None = None,
):
    """Read configuration settings and prepare simulation inputs.

    Returns:
        (star, bands, save_dir, nrho, grid_size, instrument_res, cpu_use, plot_flag,
         save_spectra, spectral_paths, phases,
         active_regions_number, active_region_box, filters, spectrum_vel_step)
    """
    # --- Read config ---
    config_file = resolve_config_file(config_file)
    logger.info("\nConfiguration: %s", config_file)
    logger.info("Run mode: %s", mode or ("debug" if debug else "simulation"))
    cfg = configparser.ConfigParser(inline_comment_prefixes=";")
    with config_file.open(encoding="utf-8") as config_stream:
        cfg.read_file(config_stream)

    # --- Path profile and save directory ---
    selected_profile = resolve_path_profile(cfg, path_profile)
    logger.info("Selected path profile: %s", selected_profile)

    save_folder_name = cfg.get("output", "save_folder_name", fallback="").strip()

    save_dir = (
        f"{save_folder_name}_{output_suffix}" if output_suffix else save_folder_name
    )
    # Append a profile name only when a named path profile is selected.
    if selected_profile != "default":
        save_dir = f"{save_dir}_{selected_profile}"

    # --- Spectroscopic bands ---
    spectral_paths = resolve_spectral_paths(cfg, selected_profile)
    bands, band_diagnostics, band_library_paths = build_spectroscopic_bands(
        cfg, spectral_paths
    )

    # Warn about narrow windows before preparing spectra.
    for band_name, params in bands.items():
        _, half_width, _, _, source = params
        if source in ("phx", "synmast") and half_width <= 50:
            logger.warning(
                "WARNING: Band %s has a window half-width of %.1f Angstroms "
                "(total width: %.1f Angstroms). Narrow windows may make "
                "continuum fitting unreliable, especially for cool-star spectra "
                "with strong molecular absorption. Inspect the continuum fit "
                "and normalized spectra, and consider a wider window.",
                band_name,
                half_width,
                2 * half_width,
            )

    logger.info(
        "\n%27s: %s",
        "Enabled spectroscopic bands",
        ", ".join(bands) if bands else "none",
    )

    filters, photometry_filters = _setup_photometry(cfg, bands, logger)
    if "photo" in bands:
        # Calculate photometry first; preserve the order of the spectral bands.
        bands = {
            "photo": bands["photo"],
            **{name: params for name, params in bands.items() if name != "photo"},
        }
    if not bands:
        raise ValueError("Enable at least one spectroscopic band or a photometry filter.")

    # --- Output & model params ---
    plot_flag = cfg.getint("output", "make_plots")
    make_gif_animation = cfg.getboolean("output", "make_gif_animation", fallback=False)
    save_spectra = cfg.getint("output", "save_spectra")
    if debug and mode not in (None, "debug"):
        raise ValueError("debug cannot be combined with another mode.")
    options = resolve_output_options(
        mode or ("debug" if debug else "simulation"), save_spectra,
        plot_flag, make_gif_animation,
    )
    plot_flag = int(options.make_plots or options.make_gif_animation)
    save_spectra = int(options.save_spectra)
    grid_size = cfg.getint("main", "grid")
    nrho = cfg.getint("main", "nrho")
    instrument_res = cfg.getint("main", "instrument_reso")

    cpu_use = cfg.getint("main", "cpu")

    # Target samples per instrumental FWHM, set in [advanced].
    spec_oversample_factor = cfg.getint(
        "advanced", "spec_oversample_factor", fallback=14
    )
    spectrum_vel_step = _calculate_velocity_step(
        instrument_res, spec_oversample_factor, logger
    )

    star, active_region_box = _setup_star_and_regions(cfg, bands, config_file, logger)
    active_regions_number = len(active_region_box[0])
    if "photo" in bands:
        star.filters = filters
        star._pyphot_filters = photometry_filters

    star.inst_res = instrument_res

    star.band_diagnostics = band_diagnostics
    star.band_library_paths = band_library_paths

    star.gaussian_parameters = {}
    star.gaussian_brightness = {}
    for band, params in bands.items():
        if params[4] == "gaussian_profile":
            star.gaussian_parameters[band] = read_gaussian_parameters(
                cfg, f"band:{band}"
            )
            star.gaussian_brightness[band] = read_gaussian_brightness(
                cfg, f"band:{band}", active_regions_number
            )

    assign_model_identity(star, active_region_box, grid_size)
    star.output_options = options
    star.save_ccf_template_metadata = options.save_intermediate
    _create_output_directories(save_dir, options)
    if not options.in_memory:
        register_model_identity(Path("outputs") / save_dir, star)
    if options.save_intermediate:
        with (Path("outputs") / save_dir / f"configuration_id-{star.model_id}.cfg").open("w", encoding="utf-8") as stream:
            cfg.write(stream)

    phases = _read_phases(cfg)
    logger.info("%27s: %d", "Rotation phases", len(phases))

    # --- PHOENIX paths ---
    star.spectral_paths = spectral_paths
    if any(p[4] != "gaussian_profile" for p in bands.values()):
        phx_root = spectral_io.require_spectral_path(spectral_paths, "phoenix_root")
        (phx_root / "SpecInt").mkdir(parents=True, exist_ok=True)
        (phx_root / "SpecFlux").mkdir(parents=True, exist_ok=True)

    # --- Projected disk grid check ---
    stsp.grid_resolution_check(grid_size)

    star.simulation_config_file = str(Path(config_file).resolve())
    rv_template_temperature = _configure_rv_template(star, cfg, config_file)

    # Prepare shared spectral inputs before constructing RV templates.
    star.prepare_spectral_inputs(
        rv_template_temperature=rv_template_temperature
    )
    star.prepare_rv_templates(
        spectrum_vel_step,
        int(options.save_intermediate),
        rv_template_temperature=rv_template_temperature,
        savehdf=options.save_intermediate,
    )

    return (
        star,
        bands,
        save_dir,
        nrho,
        grid_size,
        instrument_res,
        cpu_use,
        plot_flag,
        save_spectra,
        spectral_paths,
        phases,
        active_regions_number,
        active_region_box,
        filters,
        spectrum_vel_step,
    )
