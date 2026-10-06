"""Band selection and velocity-grid preparation for synthetic spectra.

The routines preserve existing normalization, interpolation, edge trimming,
plotting, and output formats. Source readers and exports belong to
spectral_io.

Original spectral preparation code was adapted from:
https://github.com/YinanZhao21/SOAP_GPU/blob/main/SOAP_code/phoenix_extraction.py
"""

from os import PathLike

import h5py
import matplotlib.pyplot as plt

from .visualization import PLOT_STYLE
import numpy as np
import pandas as pd
from scipy import interpolate

from .spectral_data import NativeSpectrum


def prepare_flux_spectrum(
    spectrum_source,
    spectrum_input,
    output_path,
    wavelength_min,
    wavelength_max,
    spectrum_vel_step,
    plot,
    flatten=True,
    savehdf=None,
):
    """Prepare flux on a uniform velocity grid for one simulation band.

    spectrum_input is a NativeSpectrum or a PHX/SYNMAST HDF path.
    Wavelength limits are in Angstroms; spectrum_vel_step is in km/s.
    Simulation flux uses flatten=False and retains its absolute scale.
    PHX RV templates use flatten=True with a RASSINE HDF containing
    wave, flux, and continuum_cubic columns.

    Trim 200 samples from each edge only when the resampled grid has
    more than 400 samples. Return columns v, wavelength_angstrom, and f.
    savehdf enables HDF output; plot=1 also saves a PNG when savehdf is enabled.
    """

    # ---------- Load ----------
    source = spectrum_source.lower()
    if isinstance(spectrum_input, NativeSpectrum):
        if (
            source not in ("phx", "synmast")
            or spectrum_input.source != source
            or spectrum_input.representation != "flux"
        ):
            raise ValueError("Expected a native flux spectrum matching spectrum_source.")
        if flatten:
            raise ValueError(
                "Native flux has no normalization continuum. "
                "For PHX flattening, use a RASSINE HDF containing flux "
                "and continuum_cubic columns."
            )
        wavelength = spectrum_input.wavelength.astype(np.float32, copy=False)
        flux = spectrum_input.values.astype(np.float64, copy=False)
    elif source == "phx":
        flux_table = pd.read_hdf(spectrum_input)
        wavelength = flux_table["wave"].values.astype(np.float32, copy=False)
        if flatten:
            flux = (flux_table["flux"].values / flux_table["continuum_cubic"].values).astype(
                np.float32, copy=False
            )
        else:
            flux = flux_table["flux"].values.astype(np.float64, copy=False)

    elif source == "synmast":
        # Read raw HDF5 datasets; avoid pandas overhead.
        with h5py.File(spectrum_input, "r") as handle:
            wavelength = handle["wave"][:].astype(np.float32, copy=False)
            flux = handle["flux"][:].astype(
                np.float64, copy=False
            )  # shape (n_wv, [n_mu] or 1)

    else:
        raise ValueError(f"Unknown spectrum_source: {spectrum_source}")

    # ---------- Band cut ----------
    # Wavelength limits include both endpoints.
    selected = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
    band_wavelength = wavelength[selected]
    band_flux = flux[selected]
    # A cubic spline requires at least four wavelength samples.
    if band_wavelength.size < 4:
        raise ValueError(
            f"Too few points in [{wavelength_min},{wavelength_max}] to resample flux "
            f"(got {band_wavelength.size}; need at least 4)."
        )

    # ---------- Wavelength → velocity ----------
    # Center velocity at the middle of the wavelength interval.
    c_kms = 299_792.458
    band_center = 0.5 * (wavelength_min + wavelength_max)
    band_velocity = c_kms * (band_wavelength - band_center) / band_center  # km/s

    # ---------- Resample to uniform velocity grid ----------
    # Use the smaller velocity extent to stay within the input on both sides.
    velocity_min = float(np.min(band_velocity))
    velocity_max = float(np.max(band_velocity))
    velocity_half_range = min(-velocity_min, velocity_max)  # symmetric half-span about 0
    # Step from the negative extent; the last point stays within the positive extent.
    n_velocity_bins = int(np.floor((2.0 * velocity_half_range) / spectrum_vel_step)) + 1
    velocity_grid = (-velocity_half_range) + spectrum_vel_step * np.arange(
        n_velocity_bins, dtype=np.float64
    )

    # Spline & evaluate
    spline = interpolate.splrep(band_velocity, band_flux, s=0)
    resampled_flux = interpolate.splev(velocity_grid, spline, der=0)

    # Trim edges to mitigate spline ringing (keep center region)
    trim = 200
    if velocity_grid.size > 2 * trim:
        velocity_grid = velocity_grid[trim:-trim]
        resampled_flux = resampled_flux[trim:-trim]

    # ---------- Save ----------
    prepared_spectrum = pd.DataFrame(
        {
            "v": np.asarray(velocity_grid, dtype=np.float32),
            "wavelength_angstrom": band_center * (
                1.0 + np.asarray(velocity_grid, dtype=np.float32).astype(np.float64) / c_kms
            ),
            "f": np.asarray(resampled_flux, dtype=np.float64),
        }
    )

    if savehdf:
        prepared_spectrum.to_hdf(output_path, key="spec", complevel=5, index=False)

        if plot == 1:
            with plt.rc_context(PLOT_STYLE):
                figure, (wavelength_axis, velocity_axis) = plt.subplots(
                    2, 1, figsize=(6, 4), facecolor="white", dpi=200
                )

                wavelength_axis.plot(band_wavelength, band_flux, "-", lw=1)
                velocity_axis.plot(velocity_grid, resampled_flux, "-", lw=1)

                wavelength_axis.set_xlabel("Wavelength (AA)", fontsize="small")
                velocity_axis.set_xlabel("Velocity (km/s)", fontsize="small")
                wavelength_axis.set_ylabel("Flux", fontsize="small")
                velocity_axis.set_ylabel("Flux", fontsize="small")

                if flatten:
                    wavelength_axis.set_ylim(0, 1.1)
                    velocity_axis.set_ylim(0, 1.1)
                figure.savefig(str(output_path).replace(".h5", ".png"))
                plt.close(figure)

    return prepared_spectrum


def prepare_intensity_spectrum(
    spectrum_source,
    spectrum_input,
    output_path,
    wavelength_min,
    wavelength_max,
    spectrum_vel_step,
    plot,
    savehdf=None,
    trim_edges=True,
):
    """Prepare unnormalized intensity I(wavelength, mu) for one band.

    Accept a NativeSpectrum, an HDF path, or a wave/mu/flux dictionary.
    PHX native arrays use (n_wave, n_mu); its HDF/dictionary arrays use
    (n_mu, n_wave). Reverse PHX mu and intensity columns together.
    SYNMAST arrays use (n_wave, n_mu) and retain their original mu order.

    Wavelengths are in Angstroms. PHX uses the largest native velocity
    spacing; SYNMAST uses spectrum_vel_step in km/s. trim_edges removes
    200 samples at each end. Low-resolution PHX callers disable trimming.
    savehdf controls output only; plot=1 also saves an intensity preview.
    """
    source = spectrum_source.lower()
    if source not in ("phx", "synmast"):
        raise ValueError(f"Unknown spectrum_source: {spectrum_source}")

    # Read the input independently of whether the output will be saved.
    if isinstance(spectrum_input, NativeSpectrum):
        if (
            spectrum_input.source != source
            or spectrum_input.representation != "intensity"
            or spectrum_input.mu is None
        ):
            raise ValueError("Expected an intensity spectrum with matching source and mu.")
        wavelength = spectrum_input.wavelength
        mu = spectrum_input.mu
        intensity = spectrum_input.values
    else:
        if isinstance(spectrum_input, (str, PathLike)):
            with h5py.File(spectrum_input, "r") as handle:
                wavelength = handle["wave"][:]
                mu = handle["mu"][:]
                intensity = handle["flux"][:]
        else:
            wavelength = spectrum_input["wave"][:]
            mu = spectrum_input["mu"][:]
            intensity = spectrum_input["flux"][:]
        if source == "phx":
            # PHX HDF/dictionary storage has wavelength along the second axis.
            intensity = intensity.T

    wavelength = wavelength.astype(np.float32, copy=False)
    mu = mu.astype(np.float32, copy=False)
    intensity = intensity.astype(np.float64, copy=False)
    if source == "phx":
        # Reverse mu and its matching intensity columns together.
        mu = mu[::-1]
        intensity = intensity[:, ::-1]
    elif intensity.ndim == 1:
        intensity = intensity[:, None]

    # Select the band, including both wavelength limits.
    selected = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
    band_wavelength = wavelength[selected]
    band_intensity = intensity[selected, :]
    if band_wavelength.size < 10:
        raise ValueError(
            f"Too few points in [{wavelength_min},{wavelength_max}] to resample "
            f"(got {band_wavelength.size})."
        )

    # Center velocity at the midpoint of the wavelength window.
    c_kms = 299792.458
    band_center = 0.5 * (wavelength_min + wavelength_max)
    band_velocity = c_kms * (band_wavelength - band_center) / band_center
    if source == "phx":
        # Retain PHX's coarse native sampling, independent of the requested step.
        velocity_step = (
            float(np.max(np.diff(band_velocity)))
            if band_velocity.size > 1 else spectrum_vel_step
        )
    else:
        velocity_step = float(spectrum_vel_step)

    # Stay within the input velocity extent on both sides of zero.
    velocity_min = float(np.min(band_velocity))
    velocity_max = float(np.max(band_velocity))
    velocity_half_range = min(-velocity_min, velocity_max)
    n_velocity_bins = int(np.floor((2.0 * velocity_half_range) / velocity_step)) + 1
    velocity_grid = (-velocity_half_range) + velocity_step * np.arange(
        n_velocity_bins, dtype=np.float64
    )

    # Resample each viewing-angle column independently.
    n_mu = band_intensity.shape[1]
    resampled_intensity = np.empty((velocity_grid.size, n_mu), dtype=np.float64)
    for mu_index in range(n_mu):
        spline = interpolate.splrep(band_velocity, band_intensity[:, mu_index], s=0)
        resampled_intensity[:, mu_index] = interpolate.splev(velocity_grid, spline, der=0)

    trim = 200
    if trim_edges:
        if velocity_grid.size <= 2 * trim:
            raise ValueError(
                "Intensity velocity grid must contain more than 400 samples "
                "to trim 200 samples from each edge."
            )
        velocity_grid = velocity_grid[trim:-trim]
        resampled_intensity = resampled_intensity[trim:-trim, :]

    # Match wavelengths to the stored float32 velocity bins.
    stored_velocity = velocity_grid.astype(np.float32, copy=False)
    stored_intensity = resampled_intensity.astype(np.float64, copy=False)
    columns = {
        "v": stored_velocity,
        "wavelength_angstrom": band_center * (
            1.0 + stored_velocity.astype(np.float64) / c_kms
        ),
    }
    for mu_index in range(n_mu):
        columns[f"mu_{mu[mu_index]}"] = stored_intensity[:, mu_index]
    prepared_spectrum = pd.DataFrame(columns)

    if savehdf:
        prepared_spectrum.to_hdf(output_path, key="spec", complevel=5, index=False)
        if plot == 1:
            with plt.rc_context(PLOT_STYLE):
                figure, (wavelength_axis, velocity_axis) = plt.subplots(
                    2, 1, figsize=(6, 4), facecolor="white", dpi=200
                )
                for mu_index in range(n_mu):
                    label = f"mu={mu[mu_index]}"
                    wavelength_axis.plot(
                        band_wavelength, band_intensity[:, mu_index], "-", lw=1, label=label
                    )
                    velocity_axis.plot(
                        velocity_grid, resampled_intensity[:, mu_index], "-", lw=1, label=label
                    )
                wavelength_axis.legend(fontsize="xx-small", loc="upper right")
                velocity_axis.legend(fontsize="xx-small", loc="upper right")
                wavelength_axis.set_xlabel("Wavelength (AA)", fontsize="small")
                velocity_axis.set_xlabel("Velocity (km/s)", fontsize="small")
                wavelength_axis.set_ylabel("Intensity", fontsize="small")
                velocity_axis.set_ylabel("Intensity", fontsize="small")
                figure.savefig(str(output_path).replace(".h5", ".png"))
                plt.close(figure)
    return prepared_spectrum
