"""Native spectra and prepared spectra for surface integration."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NativeSpectrum:
    """A spectrum on its native wavelength grid, independent of storage.

    Flux values have shape (n_wave,); intensity values have shape
    (n_wave, n_mu), with columns aligned to ``mu``. Readers retain their
    existing numerical precision. Units and normalization must be stated
    explicitly; an unknown values unit is represented by ``None``.
    """

    wavelength: np.ndarray
    values: np.ndarray
    representation: str
    source: str
    mu: np.ndarray | None = None
    wavelength_unit: str = 'Angstrom'
    wavelength_frame: str = 'vacuum'
    values_unit: str | None = None
    normalization: str = 'none'
    metadata: dict[str, Any] = field(default_factory=dict)


class PreparedSpectrum:
    """Hold a local spectrum and velocity-grid metadata for surface integration.

    Arrays reference the prepared spectral storage on Star. The spectrum is
    not padded here; integration kernels use n_padded_bins and vel_half_range
    to allocate the output grid and place Doppler-shifted contributions.

    Attributes
    ----------
    spectrum_vel_grid : ndarray, shape (n_spectrum_bins,)
        Input velocity coordinates in km/s; expected to be uniform and increasing.
    wavelength_grid : ndarray, shape (n_spectrum_bins,)
        Resampled wavelengths in Angstroms, aligned with spectrum_vel_grid.
    spectrum : ndarray
        Flux values with shape (n_spectrum_bins,), or specific intensities
        with shape (n_spectrum_bins, n_mu).
    mu_grid : ndarray, shape (n_mu,)
        Viewing-angle cosines; present only in intensity mode.
    spectrum_vel_half_range : float
        Maximum input velocity in km/s; the half-width for a symmetric grid.
    vel_step : float
        Input velocity spacing in km/s.
    n_spectrum_bins : int
        Number of input spectral bins.
    n_padded_bins : int
        Number of output bins including velocity padding.
    vel_half_range : float
        Half-width of the padded output velocity grid in km/s.
    """

    def __init__(self, star, band, temperature, spectrum_mode, spectrum_source):
        """Select a prepared spectrum and determine its output padding.

        Parameters
        ----------
        star : Star
            Stellar rotation velocity and prepared flux/intensity storage.
        band : str
            Configured band name used in spectral lookup keys.
        temperature : float
            Component temperature in Kelvin, formatted to four decimals for lookup.
        spectrum_mode : str
            Spectral mode: 'flux' or 'intensity'.
        spectrum_source : str
            Library or profile label used in storage keys, such as 'phx',
            'synmast'.
        """
        if spectrum_mode == "flux":
            storage = star.flux_spec
        elif spectrum_mode == "intensity":
            storage = star.int_spec
        else:
            raise ValueError(f"Unsupported spectrum mode: {spectrum_mode!r}")

        # Reference the prepared arrays without copying them.
        prefix = f"{band}_{spectrum_source}_{spectrum_mode}_{temperature:1.4f}"
        self.spectrum_vel_grid = storage[f"{prefix}_v"]
        self.wavelength_grid = storage[f"{prefix}_w"]
        self.spectrum = storage[f"{prefix}_f"]
        if spectrum_mode == "intensity":
            self.mu_grid = storage[f"mu_{spectrum_source}"]

        if not (
            self.wavelength_grid.shape == self.spectrum_vel_grid.shape
            and self.spectrum.shape[0] == self.spectrum_vel_grid.size
        ):
            raise ValueError("Wavelength, velocity and spectrum bins must align.")

        if self.spectrum_vel_grid.ndim != 1 or self.spectrum_vel_grid.size < 2:
            raise ValueError("Velocity grid must be one-dimensional with at least two points.")
        self.vel_step = self.spectrum_vel_grid[1] - self.spectrum_vel_grid[0]
        if not np.isfinite(self.vel_step) or self.vel_step <= 0:
            raise ValueError("Velocity spacing must be finite and positive.")
        self.spectrum_vel_half_range = np.max(self.spectrum_vel_grid)
        self.n_spectrum_bins = len(self.spectrum_vel_grid)

        # Allow for rotational shifts, with at least 20 km/s padding on each side.
        padding_vel = max(1.1 * star.rotation_vel, 20.0)
        n_padded_bins_estimate = self.n_spectrum_bins + round(
            (padding_vel / self.vel_step) * 2
        )

        # Add an even number of bins to preserve the input grid's parity.
        # An odd input grid therefore retains a central bin in the padded grid.
        if np.round((padding_vel / self.vel_step) * 2) % 2 == 0:
            self.n_padded_bins = int(n_padded_bins_estimate)
        else:
            self.n_padded_bins = int(n_padded_bins_estimate + 1)

        # The padded grid has n_padded_bins - 1 intervals at the original spacing.
        self.vel_half_range = self.vel_step * (self.n_padded_bins - 1) / 2.0
