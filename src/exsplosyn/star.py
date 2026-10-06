"""Stellar parameters and methods for preparing local spectra."""

import numpy as np
from ldtk import BoxcarFilter, LDPSetCreator
from ldtk.ldtk import LDPSet

from . import spectral_setup, spectral_io
from .gaussian_profiles import build_gaussian_profile


# Avoid interpolation errors at the mu=0 endpoint.
def _resample_linear_mu_safe(self, nmu=50, eps=1e-6):
    self.resample(mu=np.linspace(eps, 1.0, nmu))


class Star:
    """Store stellar parameters and prepare local spectra for surface integration.

    Simulation setup attaches spectral paths and band-specific settings before
    the preparation methods are called. Temperatures are stored in Kelvin.
    """

    def __init__(
        self,
        prot,
        rotation_vel,
        inclination_angle,
        rad,
        photosphere_temperature,
        T_cool,
        T_hot,
        logg,
        metallicity,
        bands,
    ):
        """Initialize the stellar geometry and component temperatures.

        Parameters
        ----------
        prot : float
            Rotation period in days.
        rotation_vel : float
            Equatorial rotation velocity in km/s.
        inclination_angle : float
            Rotation-axis inclination angle to the line of sight, in degrees.
        rad : float
            Stellar radius in km, already converted from solar-radius units.
        photosphere_temperature : float
            Quiet photospheric temperature in Kelvin.
        T_cool : array_like
            Region temperatures below the photosphere temperature, in Kelvin.
        T_hot : array_like
            Region temperatures above the photosphere temperature, in Kelvin.
        logg : float
            Base-10 logarithm of surface gravity in cgs units.
        metallicity : float
            Metallicity [Fe/H].
        bands : dict
            Band names mapped to their spectral and diagnostic settings.
        """
        self.prot = prot
        self.rotation_vel = rotation_vel
        self.inclination_angle = inclination_angle
        self.rad = rad
        self.photosphere_temperature = np.round(photosphere_temperature, 4)
        self.T_cool = np.round(T_cool, 4)
        self.T_hot = np.round(T_hot, 4)
        self.logg = logg
        self.metallicity = metallicity
        self.bands = bands

    def prepare_quadratic_limb_darkening(
        self, temperatures, band_names, band_centers, window_half_widths
    ):
        """Compute quadratic limb darkening for each temperature and wavelength band.

        Fit LDTk intensity profiles using boxcar bands and the stellar gravity
        and metallicity. Store coefficients (a1, a2) for the law
        I(mu)/I(1) = 1 - a1*(1-mu) - a2*(1-mu)**2 in self.LD_coeff.

        Parameters
        ----------
        temperatures : array_like
            Component temperatures in Kelvin.
        band_names : sequence of str
            Unique band names, in the same order as centers and half-widths.
        band_centers : array_like
            Band centers in Angstroms.
        window_half_widths : array_like
            Band half-widths in Angstroms, matching band_centers.

        Notes
        -----
        Requires self.spectral_paths for the PHOENIX intensity cache location.
        Keys use the supplied band names and temperatures rounded to four decimals.
        Sampling uses seed 69 and restores the caller's NumPy random state.
        """
        # Accept scalar or array inputs for band geometry.
        band_centers = np.atleast_1d(np.array(band_centers, dtype=float))
        window_half_widths = np.atleast_1d(np.array(window_half_widths, dtype=float))
        band_names = [band_names] if isinstance(band_names, str) else list(band_names)
        if band_centers.ndim != 1 or window_half_widths.ndim != 1:
            raise ValueError("Band centers and half-widths must be one-dimensional.")
        if not (len(band_names) == len(band_centers) == len(window_half_widths)):
            raise ValueError(
                "Band names, centers and half-widths must have matching lengths."
            )
        if any(not isinstance(name, str) or not name.strip() for name in band_names):
            raise ValueError("Band names must be nonempty strings.")
        if len(set(band_names)) != len(band_names):
            raise ValueError("Band names must be unique.")

        # Convert Å -> nm for ldtk
        lc_nm = band_centers / 10.0
        wid_nm = window_half_widths / 10.0

        # Define boxcar edges as center +/- half-width, in nm.
        filters = [
            BoxcarFilter(band_names[i], lc_nm[i] - wid_nm[i], lc_nm[i] + wid_nm[i])
            for i in range(len(lc_nm))
        ]

        limb_darkening_coefficients = {}
        # Sample input uncertainties to construct the intensity profiles.
        nsamples = 2000

        random_state = np.random.get_state()
        try:
            np.random.seed(69)
            for component_temperature in np.atleast_1d(temperatures).astype(float):
                # Supply nonzero uncertainties for LDTk profile sampling.
                profile_creator = LDPSetCreator(
                    teff=(component_temperature, 10.0),
                    logg=(self.logg, 0.01),
                    z=(self.metallicity, 0.01),
                    filters=filters,
                    dataset="visir-lowres",
                    cache=spectral_io.require_spectral_path(
                        self.spectral_paths, "phoenix_intensity_root"
                    ),
                )

                # Use the safe grid only while LDTk constructs the profiles.
                original_resampling = LDPSet.resample_linear_mu
                try:
                    LDPSet.resample_linear_mu = _resample_linear_mu_safe
                    profiles = profile_creator.create_profiles(nsamples=nsamples)
                finally:
                    LDPSet.resample_linear_mu = original_resampling

                # Get quadratic coefficients: returns (means, covariance)
                quadratic_coefficients, _ = profiles.coeffs_qd(
                    do_mc=False
                )  # shape: (n_filters, 2) for (u1, u2)

                # Use configured band names directly; wavelength-derived aliases can collide.
                for band_name, coefficients in zip(band_names, quadratic_coefficients):
                    key = f"{band_name}_{np.round(component_temperature, 4)}"
                    limb_darkening_coefficients[key] = coefficients
        finally:
            np.random.set_state(random_state)

        self.LD_coeff = limb_darkening_coefficients

    def prepare_flux_spectra(self, spectrum_vel_step, plot, savehdf=None):
        """Prepare local flux spectra for configured flux bands.

        Delegate library reading and preprocessing to spectral_setup, using
        the photosphere, cool and hot component temperatures. Store the
        prepared spectra in self.flux_spec; Gaussian profiles are built separately.

        Parameters
        ----------
        spectrum_vel_step : float
            Velocity sampling interval in km/s.
        plot : bool
            Whether to generate preprocessing plots.
        savehdf : bool or None
            Whether to save prepared spectra; None leaves this disabled.
        """
        return spectral_setup.prepare_flux_spectra(
            self, spectrum_vel_step, plot, savehdf=savehdf
        )

    def prepare_gaussian_profiles(self):
        """Build local Gaussian profiles for configured Gaussian bands."""
        self.gaussian_profiles = {
            band: build_gaussian_profile(parameters)
            for band, parameters in getattr(self, "gaussian_parameters", {}).items()
        }

    def prepare_black_region_spectra(self):
        """Add zero spectra for non-emitting regions using the photosphere grids."""
        if not getattr(self, "has_black_spots", False):
            return

        black_temp = float(self.black_spot_sentinel)
        phot_temp = float(np.round(self.photosphere_temperature, 4))

        for band_name, params in self.bands.items():
            _, _, _, spectrum_mode, spectrum_source = params
            if spectrum_source == "gaussian_profile":
                continue

            phot_prefix = (
                f"{band_name}_{spectrum_source}_{spectrum_mode}_{phot_temp:1.4f}"
            )
            black_prefix = (
                f"{band_name}_{spectrum_source}_{spectrum_mode}_{black_temp:1.4f}"
            )

            if spectrum_mode == "flux":
                spectral_storage = self.flux_spec
            elif spectrum_mode in ("intensity", "photo"):
                spectral_storage = self.int_spec
            else:
                raise ValueError(
                    f"Unsupported spectrum type for black spot: {spectrum_mode!r}"
                )

            spectral_storage[f"{black_prefix}_v"] = np.asarray(
                spectral_storage[f"{phot_prefix}_v"]
            ).copy()
            spectral_storage[f"{black_prefix}_w"] = np.asarray(
                spectral_storage[f"{phot_prefix}_w"]
            ).copy()
            spectral_storage[f"{black_prefix}_f"] = np.zeros_like(
                spectral_storage[f"{phot_prefix}_f"], dtype=np.float64
            )

            # Flux integration explicitly looks up quadratic LD coefficients.
            # Intensity spectra already encode their center-to-limb behavior.
            if spectrum_mode == "flux":
                phot_ld_key = f"{band_name}_{phot_temp}"
                black_ld_key = f"{band_name}_{black_temp}"

                if phot_ld_key not in self.LD_coeff:
                    raise KeyError(
                        f"Cannot create black-spot LD entry because "
                        f"{phot_ld_key!r} is missing."
                    )

                self.LD_coeff[black_ld_key] = np.asarray(
                    self.LD_coeff[phot_ld_key]
                ).copy()

    def prepare_spectral_inputs(self, rv_template_temperature=None):
        """Initialize spectral storage and prepare required PHOENIX flux inputs.

        Include all component temperatures needed for PHOENIX flux simulations
        and the selected temperature needed for PHOENIX-based RV templates.
        Delegate file preparation and in-memory storage to spectral_setup.

        Parameters
        ----------
        rv_template_temperature : float or None
            RV CCF template temperature in Kelvin. None selects self.photosphere_temperature.

        Notes
        -----
        Call before preparing simulation spectra or RV templates. Library and
        tool locations are read from self.spectral_paths.
        """
        return spectral_setup.prepare_spectral_inputs(
            self, rv_template_temperature=rv_template_temperature
        )

    def prepare_rv_templates(
        self, spectrum_vel_step, plot, rv_template_temperature=None, savehdf=None
    ):
        """Prepare PHOENIX-based CCF templates for bands requesting RV.

        Delegate continuum normalization and template preprocessing to
        spectral_setup. Store templates in self.spec_flatten, separately
        from the local spectra used for surface integration.

        Parameters
        ----------
        spectrum_vel_step : float
            Velocity sampling interval in km/s.
        plot : bool
            Whether to generate preprocessing plots.
        savehdf : bool or None
            Whether to save prepared spectra; None leaves this disabled.
        rv_template_temperature : float or None
            RV CCF template temperature in Kelvin. None selects self.photosphere_temperature.

        Notes
        -----
        Call prepare_spectral_inputs first. Library and tool locations are
        read from self.spectral_paths.
        """
        return spectral_setup.prepare_rv_templates(
            self,
            spectrum_vel_step,
            plot,
            rv_template_temperature=rv_template_temperature,
            savehdf=savehdf,
        )

    def prepare_intensity_spectra(self, spectrum_vel_step, plot, savehdf=None):
        """Prepare local mu-dependent spectra for configured intensity bands.

        Delegate library reading and preprocessing to spectral_setup, using
        the photosphere, cool and hot component temperatures. Store prepared
        spectra in self.int_spec, preserving their mu dependence.

        Parameters
        ----------
        spectrum_vel_step : float
            Velocity sampling interval in km/s.
        plot : bool
            Whether to generate preprocessing plots.
        savehdf : bool or None
            Whether to save prepared spectra; None leaves this disabled.
        """
        return spectral_setup.prepare_intensity_spectra(
            self, spectrum_vel_step, plot, savehdf=savehdf
        )

    def prepare_phx2013_flux_inputs(self, temperatures, alpha=0):
        """Load PHOENIX 2013 flux spectra and export files for template preprocessing.

        Delegate to spectral_setup. Populate self.phx_flux_spectra with loaded
        spectra and self.phx_flux_file with HDF filenames used by RASSINE.

        Parameters
        ----------
        temperatures : array_like
            Temperatures in Kelvin, within the supported preprocessing range.
        alpha : float
            Alpha-element abundance enhancement [alpha/Fe].
        """
        return spectral_setup.prepare_phx2013_flux_inputs(self, temperatures, alpha=alpha)
