import os
import shutil
import logging
import tempfile
from pathlib import Path
from PIL import Image
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.transforms import ScaledTranslation

import numpy as np
from pqdm.processes import pqdm

logger = logging.getLogger(__name__)

MODEL_INFO_FONT_SIZE = 4.8
MODEL_INFO_LINE_SPACING = 7.5  # points

PLOT_STYLE = {
    "figure.dpi": 250,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.style": "normal",
    "xtick.top": True,
    "xtick.direction": "in",
    "ytick.right": True,
    "ytick.direction": "in",
}


def rotation_matrix_axis_angle(axis=[0, 1, 0], theta=0):
    """
    Rotate an arbitrary point by an axis and an angle.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """
    cost = np.cos(theta)
    sint = np.sin(theta)

    return np.reshape(
        [
            cost + axis[0] * axis[0] * (1 - cost),
            axis[0] * axis[1] * (1 - cost) - axis[2] * sint,
            axis[0] * axis[2] * (1 - cost) + axis[1] * sint,
            axis[1] * axis[0] * (1 - cost) + axis[2] * sint,
            cost + axis[1] * axis[1] * (1 - cost),
            axis[1] * axis[2] * (1 - cost) - axis[0] * sint,
            axis[2] * axis[0] * (1 - cost) - axis[1] * sint,
            axis[2] * axis[1] * (1 - cost) + axis[0] * sint,
            cost + axis[2] * axis[2] * (1 - cost),
        ],
        [3, 3],
    )


def get_ortho_latitude_lines(radius, inc=np.pi / 2, obl=0, dlat=np.pi / 6, npts=1000):
    """
    Return the lines of constant latitude on an orthographic projection.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """
    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Latitude lines
    res = []
    latlines = np.arange(-np.pi / 2, np.pi / 2, dlat)[1:]
    for lat in latlines:

        # Figure out the equation of the ellipse
        y0 = np.sin(lat) * si
        a = np.cos(lat)
        b = a * ci
        x = np.linspace(-a, a, npts)
        y1 = y0 - b * np.sqrt(1 - (x / a) ** 2)
        y2 = y0 + b * np.sqrt(1 - (x / a) ** 2)

        # Mask lines on the backside
        if si != 0:
            if inc > np.pi / 2:
                ymax = y1[np.argmax(x**2 + y1**2)]
                y1[y1 < ymax] = np.nan
                ymax = y2[np.argmax(x**2 + y2**2)]
                y2[y2 < ymax] = np.nan
            else:
                ymax = y1[np.argmax(x**2 + y1**2)]
                y1[y1 > ymax] = np.nan
                ymax = y2[np.argmax(x**2 + y2**2)]
                y2[y2 > ymax] = np.nan

        # Rotate them
        for y in (y1, y2):
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr * radius, y * radius))

    return res


def get_ortho_longitude_lines(
    radius, inc=np.pi / 2, obl=0, theta=0, dlon=np.pi / 6, npts=1000
):
    """
    Return the lines of constant longitude on an orthographic projection.
    from Starry: https://github.com/rodluger/paparazzi/blob/119743ec26bbca8b3240c38be36b0c47cb2c9880/src/figures/linalg.py
    """

    # Angular quantities
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(obl)
    so = np.sin(obl)

    # Are we (essentially) equator-on?
    equator_on = (inc > 88 * np.pi / 180) and (inc < 92 * np.pi / 180)

    # Longitude grid lines
    res = []
    if equator_on:
        offsets = np.arange(-np.pi / 2, np.pi / 2, dlon)
    else:
        offsets = np.arange(0, 2 * np.pi, dlon)

    for offset in offsets:

        # Choose which longitude branches face the observer.
        if equator_on:
            sgns = [1]
            if np.cos(theta + offset) >= 0:
                bsgn = 1
            else:
                bsgn = -1
        else:
            bsgn = 1
            if np.cos(theta + offset) >= 0:
                sgns = np.array([1, -1])
            else:
                sgns = np.array([-1, 1])

        for lon, sgn in zip([0, np.pi], sgns):

            # Viewed at i = 90
            y = np.linspace(-1, 1, npts)
            b = bsgn * np.sin(lon - theta - offset)
            x = b * np.sqrt(1 - y**2)
            z = sgn * np.sqrt(np.abs(1 - x**2 - y**2))

            if equator_on:

                pass

            else:

                # Rotate by the inclination angle
                R = rotation_matrix_axis_angle([1, 0, 0], np.pi / 2 - inc)
                v = np.vstack((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)))
                x, y, _ = np.dot(R, v)

                # Mask lines on the backside
                if si != 0:
                    if inc < np.pi / 2:
                        imax = np.argmax(x**2 + y**2)
                        y[: imax + 1] = np.nan
                    else:
                        imax = np.argmax(x**2 + y**2)
                        y[imax:] = np.nan

            # Rotate by the obliquity
            xr = -x * co + y * so
            yr = x * so + y * co
            res.append((xr * radius, yr * radius))

    return res


def add_stellar_coordinate_grid(
    axins, inclination_angle, radius, phase, *, latitude_lines=None
):
    """Overlay visible latitude and longitude lines on the stellar disk."""
    theta = phase * (2 * np.pi)

    lat_lines = latitude_lines
    if lat_lines is None:
        lat_lines = get_ortho_latitude_lines(
            radius, inc=inclination_angle * np.pi / 180
        )
    for x, y in lat_lines:
        axins.plot(x, y, "#aaaaaa", lw=0.3, alpha=0.3, zorder=11)

    lon_lines = get_ortho_longitude_lines(
        radius,
        inc=inclination_angle * np.pi / 180,
        theta=np.pi + theta,
    )
    for n, l in enumerate(lon_lines):
        axins.plot(l[0], l[1], "#aaaaaa", lw=0.3, alpha=0.3, zorder=11)
    return axins


def _phase_xlim(phases):
    """Cover all input phases with a small margin, including single-phase runs."""
    lower, upper = float(np.min(phases)), float(np.max(phases))
    margin = 0.1 * (upper - lower) if upper > lower else 0.15
    return lower - margin, upper + margin


def _prepare_phase_plot_data(star, band, rv, ewr, gaussian, photometry_for_plot, *, phases=None):
    """Set fixed diagnostic ranges from all phases once per band."""
    selected = getattr(star, "band_diagnostics", {}).get(band, ("rv",))
    show_rv = (gaussian or "rv" in selected) and np.any(np.isfinite(rv))
    show_ewr = not gaussian and "ewr_FeOH" in selected and np.any(np.isfinite(ewr))
    data = {"show_rv": show_rv, "show_ewr": show_ewr, "photo_ylim": None}
    if phases is not None:
        # Every GIF frame uses the full phase range, even while curves build up.
        data["phase_xlim"] = _phase_xlim(phases)
    if show_rv:
        values = np.asarray(rv)[np.isfinite(rv)]
        # Give constant RV curves a finite plotting range as well.
        margin = max(3 * float(np.std(values)), 0.01)
        data["rv_ylim"] = (np.min(values) - margin, np.max(values) + margin)
    if show_ewr:
        values = np.asarray(ewr)[np.isfinite(ewr)]
        data["ewr_ylim"] = (np.min(values) - 0.1, np.max(values) + 0.1)
    finite_values = []
    if photometry_for_plot is not None:
        for _, curve in photometry_for_plot["curves"][:2]:
            curve = np.asarray(curve)
            finite_values.extend(curve[np.isfinite(curve)])
    if finite_values:
        lower, upper = min(finite_values), max(finite_values)
        margin = max(0.1 * (upper - lower), 0.002)
        data["photo_ylim"] = (lower - margin, upper + margin)
    return data


def _prepare_stellar_annotation(text):
    """Measure columns once so math symbols and separators align across frames."""
    rows = [tuple(value.strip() for value in row.split("|")) for row in text.splitlines()]
    if not rows or len({len(row) for row in rows}) != 1 or len(rows[0]) == 1:
        return [(0.0, 0.0, text)]
    font = FontProperties(family="monospace", size=MODEL_INFO_FONT_SIZE)
    metrics = TextToPath()
    widths = [max(metrics.get_text_width_height_descent(value, font, ismath="$" in value)[0]
                  for value in column) for column in zip(*rows)]
    cells = []
    for row_index, row in enumerate(rows):
        offset = 0.0
        for column_index, value in enumerate(row):
            cells.append((offset, -MODEL_INFO_LINE_SPACING * row_index, value))
            if column_index < len(row) - 1:
                offset += widths[column_index] + 1.5
                cells.append((offset, -MODEL_INFO_LINE_SPACING * row_index, "|"))
                offset += 3.5
    return cells


def _prepare_region_annotation(text):
    """Place fixed region statistics and the model ID on evenly spaced lines."""
    return [(0.0, -MODEL_INFO_LINE_SPACING * index, line)
            for index, line in enumerate(text.splitlines())]


def _prepare_band_plot_data(
    star,
    grid,
    phase_spectra,
    rv,
    ewr,
    band,
    spectrum_mode,
    spectrum_source,
    photometry_for_plot,
    *,
    phases=None,
    stellar_info_text="",
    region_info_text="",
    region_brightness=None,
    surface_flux_max=None,
):
    """Prepare wavelength, limits and fixed disk geometry shared by all frames."""
    gaussian = spectrum_source == "gaussian_profile"
    if gaussian:
        wavelength = star.gaussian_profiles[band][1]
        off_disk = None
    else:
        storage = star.int_spec if spectrum_mode == "intensity" else star.flux_spec
        velocity = storage[
            f"{band}_{spectrum_source}_{spectrum_mode}_{star.photosphere_temperature:1.4f}_v"
        ]
        center = star.bands[band][0]
        wavelength = (velocity * center / 299792.458) + center
        coord = -1.0 + np.arange(grid, dtype=np.float64) * (2.0 / grid)
        Y, Z = np.meshgrid(coord, coord, indexing="ij")
        off_disk = (Y * Y + Z * Z) >= 1.0
    model_hash = getattr(star, "model_hash", "")[:10]
    if not model_hash:
        model_hash = getattr(star, "model_id", "").rpartition("_")[2]
    return {
        "model_id_text": f"Model ID: {model_hash}" if model_hash else "",
        "grid": grid,
        "inclination_angle": star.inclination_angle,
        "gaussian": gaussian,
        "phases": phases,
        "rv": rv,
        "ewr": ewr,
        "stellar_info_text": stellar_info_text,
        "stellar_annotation": _prepare_stellar_annotation(stellar_info_text),
        "region_info_text": region_info_text,
        "region_annotation": _prepare_region_annotation(region_info_text),
        "photometry": photometry_for_plot,
        "region_brightness": region_brightness,
        "surface_flux_max": surface_flux_max,
        "wavelength": wavelength,
        "spectrum_ylim": (np.min(phase_spectra) - 0.05, 1.05),
        "off_disk": off_disk,
        "latitude_lines": get_ortho_latitude_lines(
            grid // 2,
            inc=star.inclination_angle * np.pi / 180,
        ),
        "diagnostics": _prepare_phase_plot_data(
            star, band, rv, ewr, gaussian, photometry_for_plot, phases=phases
        ),
    }


def _plot_phase_diagnostics(
    ax,
    star,
    band,
    phases,
    rv,
    ewr,
    phase_index,
    gaussian,
    photometry_for_plot,
    *,
    plot_data=None,
):
    """Plot RV on the left, EWR on the right, and the first two light curves outside."""
    if plot_data is None:
        plot_data = _prepare_phase_plot_data(
            star, band, rv, ewr, gaussian, photometry_for_plot, phases=phases
        )
    show_rv, show_ewr = plot_data["show_rv"], plot_data["show_ewr"]
    axes = [ax]
    ax.set_zorder(2)
    end = phase_index + 1
    phase_xlim = plot_data.get("phase_xlim")
    ax.set_xlim(*(phase_xlim if phase_xlim is not None else _phase_xlim(phases)))
    ax.set_xlabel("Phase", size="small")
    ax.tick_params(
        axis="x", which="both", labelsize="small", top=True, direction="in", width=0.7
    )
    if show_rv:
        ax.plot(phases[:end], rv[:end], "-o", color="skyblue", ms=2, mew=2)
        ax.plot(
            phases[phase_index], rv[phase_index], "o", color="tab:blue", ms=2, mew=2
        )
        ax.set_ylim(*plot_data["rv_ylim"])
        label = "First-moment RV" if gaussian else "Spot induced RV"
        ax.set_ylabel(label + r" [km s$^{-1}$]", size="small", color="tab:blue")
        ax.tick_params(
            axis="y",
            which="both",
            colors="tab:blue",
            labelsize="small",
            direction="in",
            width=0.7,
        )
    else:
        ax.set_yticks([])

    if show_ewr:
        ewr_axis = ax.twinx()
        ewr_axis.set_zorder(1)
        axes.append(ewr_axis)
        ewr_axis.plot(phases[:end], ewr[:end], "-o", color="tab:grey", ms=2, mew=2)
        ewr_axis.plot(
            phases[phase_index], ewr[phase_index], "o", color="0.3", ms=2, mew=2
        )
        ewr_axis.set_ylim(*plot_data["ewr_ylim"])
        ewr_axis.set_ylabel(
            "Equivalent Width Ratio (OH/Fe)",
            size="small",
            color="tab:grey",
            rotation=270,
            labelpad=14,
        )
        ewr_axis.tick_params(
            axis="y",
            which="both",
            colors="tab:grey",
            labelsize="small",
            direction="in",
            width=0.7,
        )

    curves = [] if photometry_for_plot is None else photometry_for_plot["curves"][:2]
    if curves:
        photo_axis = ax.twinx()
        # Draw light curves behind RV/EWR, including across separate axes.
        photo_axis.set_zorder(0)
        axes.append(photo_axis)
        photo_axis.spines["right"].set_position(("axes", 1.18))
        photo_phases = photometry_for_plot["phases"]
        for (name, curve), color in zip(curves, ("tab:red", "tab:purple")):
            curve = np.asarray(curve)
            photo_axis.plot(
                photo_phases[:end],
                curve[:end],
                "-",
                color=color,
                marker="",
                lw=1,
                zorder=0,
                label=name.rsplit("/", 1)[-1],
            )
        if plot_data["photo_ylim"] is not None:
            photo_axis.set_ylim(*plot_data["photo_ylim"])
        photo_axis.set_ylabel(
            "Relative flux", size="small", color="tab:red", rotation=270, labelpad=14
        )
        photo_axis.tick_params(
            axis="y",
            which="both",
            colors="tab:red",
            labelsize="small",
            direction="in",
            width=0.7,
        )
        photo_axis.legend(loc="upper left", fontsize="xx-small", labelcolor="linecolor")
    if len(axes) > 1:
        # Transparent foreground axes keep the background light curves visible.
        ax.patch.set_visible(False)
    return axes


@mpl.rc_context(PLOT_STYLE)
def save_lightcurve_plot(phases, filters, relative_flux_curves, output_path):
    """Save a relative-flux plot containing all configured photometry filters."""
    phases = np.asarray(phases)
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
    try:
        # Show the interior phases across three rotation cycles.
        for filter_index, filter_name in enumerate(filters):
            curve = relative_flux_curves[filter_index][1:-1]
            line = ax.plot(
                phases[1:-1] - 1,
                curve,
                label=filter_name.split("/")[1],
                marker="o",
                ms=2,
                alpha=0.7,
            )
            for phase_offset in (0, 1):
                ax.plot(
                    phases[1:-1] + phase_offset,
                    curve,
                    label="__",
                    marker="o",
                    ms=2,
                    alpha=0.7,
                    color=line[0].get_color(),
                )
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlabel("Phase", size="small")
        ax.set_ylabel("Relative flux", size="small")
        ax.legend(
            fontsize="xx-small",
            edgecolor="white",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)


@mpl.rc_context(PLOT_STYLE)
def plot_simulation_frame(
    star,
    grid,
    phases,
    phase_spectra,
    ccf_rv_shift,
    quite_surface_2d,
    outdir,
    ohfe_ratio,
    stellar_info_text,
    region_info_text,
    band,
    spectrum_mode,
    spectrum_source,
    j,
    *,
    region_cube=None,
    region_brightness=None,
    region_fspot=None,
    surface_flux_max=None,
    frame_directory=None,
    photometry_for_plot=None,
    output_path=None,
    plot_data=None,
):
    """Draw the stellar disk, spectrum and diagnostic curves through one phase."""
    gaussian = spectrum_source == "gaussian_profile"
    if plot_data is None:
        plot_data = _prepare_band_plot_data(
            star,
            grid,
            phase_spectra,
            ccf_rv_shift,
            ohfe_ratio,
            band,
            spectrum_mode,
            spectrum_source,
            photometry_for_plot,
            stellar_info_text=stellar_info_text,
            region_info_text=region_info_text,
        )
    plot_data = dict(plot_data)
    if plot_data["stellar_info_text"] != stellar_info_text:
        plot_data["stellar_annotation"] = _prepare_stellar_annotation(stellar_info_text)
    if plot_data["region_info_text"] != region_info_text:
        plot_data["region_annotation"] = _prepare_region_annotation(region_info_text)
    if not gaussian and surface_flux_max is None:
        surface_flux_max = float(np.nanmax(quite_surface_2d))
    plot_data.update(
        phases=phases,
        rv=ccf_rv_shift,
        ewr=ohfe_ratio,
        stellar_info_text=stellar_info_text,
        region_info_text=region_info_text,
        region_brightness=region_brightness,
        surface_flux_max=surface_flux_max,
    )
    frame_data = _prepare_frame_data(
        star,
        j,
        phases,
        phase_spectra,
        quite_surface_2d,
        gaussian=gaussian,
        region_cube=region_cube,
        region_fspot=region_fspot,
    )
    _plot_prepared_frame(
        frame_data,
        plot_data,
        frame_directory=frame_directory or Path(outdir).parent / "gif_tmp",
        output_path=output_path,
    )


def _prepare_frame_data(
    star,
    index,
    phases,
    spectra,
    surface_flux,
    *,
    gaussian,
    region_cube=None,
    region_fspot=None,
):
    """Select one phase's arrays and scalar labels; do not copy the full cubes."""
    frame = {
        "index": index,
        "phase": phases[index],
        "spectrum": spectra[index],
        "map": region_cube[index] if gaussian else surface_flux[index],
        "active_filling_f": (
            region_fspot[index] if gaussian else star._phase_fspot[index]
        ),
    }
    if not gaussian:
        frame.update(
            apparent_teff=star._phase_teff[index],
            cool_filling_f=star._phase_fcool[index],
            hot_filling_f=star._phase_fhot[index],
        )
    return frame


@mpl.rc_context(PLOT_STYLE)
def _plot_prepared_frame(
    frame_data, plot_data, *, frame_directory=None, output_path=None
):
    """Render one phase using only its arrays and the band's small plot context."""
    j = frame_data["index"]
    gaussian = plot_data["gaussian"]
    grid = plot_data["grid"]
    phases = plot_data["phases"]
    photometry_for_plot = plot_data["photometry"]
    region_brightness = plot_data["region_brightness"]
    _fspot = frame_data["active_filling_f"] * 100
    if not gaussian:
        _teff = frame_data["apparent_teff"]
        _fcool = frame_data["cool_filling_f"] * 100
        _fhot = frame_data["hot_filling_f"] * 100

    # -------------------------------------------------
    show_photometry = photometry_for_plot is not None and bool(
        photometry_for_plot["curves"]
    )
    fig = plt.figure(figsize=(7.5 if show_photometry else 7, 3.5), dpi=300)

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.0, right=0.38, bottom=0.28, top=1.0)
    ax1 = plt.subplot(gs1[0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(
        left=0.46,
        right=0.88 if show_photometry else 1.0,
        bottom=0.0,
        top=1.0,
        wspace=0.0,
        hspace=0.25,
    )
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs2[1])

    radius = grid // 2

    if gaussian:
        # IDs use [iy, iz]; transpose for sky-plane display. Mask -1 before lookup.
        ids = frame_data["map"].T
        st_spot_surface = np.full(ids.shape, np.nan)
        disk = ids >= 0
        st_spot_surface[disk] = region_brightness[ids[disk]]
        # Fix the color scale across phases; retain bright values above one.
        surface_vmax = max(1.0, float(np.max(region_brightness)))
    else:
        # --- true geometry masking ---
        surface_flux = np.asarray(
            frame_data["map"],
            dtype=np.float64,
        )
        # Band-integrated surface flux, scaled by one maximum across all phases.
        # This is a relative flux map, not a temperature map.
        surface_max = plot_data["surface_flux_max"]

        if not np.isfinite(surface_max) or surface_max <= 0.0:
            raise ValueError(f"Invalid shared surface-flux maximum: " f"{surface_max}")

        st_spot_surface = surface_flux / surface_max

        # Remove tiny negative values from floating-point subtraction,
        # while preserving genuine zero-flux black pixels.
        st_spot_surface = np.clip(
            st_spot_surface,
            0.0,
            1.0,
        )

        # Reuse the geometry mask without changing the original surface cube.
        st_spot_surface[plot_data["off_disk"]] = np.nan

        surface_vmax = 1.0

    # Surface maps are displayed as [iz, iy], placing y horizontally.
    ax1.imshow(
        st_spot_surface,
        extent=[-radius, radius, -radius, radius],
        cmap="Oranges_r",
        vmin=0,
        vmax=surface_vmax,
        origin="lower",
    )
    ax1 = add_stellar_coordinate_grid(
        ax1,
        plot_data["inclination_angle"],
        radius,
        frame_data["phase"],
        latitude_lines=plot_data["latitude_lines"],
    )

    ax1.text(
        0.5,
        -0.07,
        rf"Phase ${frame_data['phase']:.3f}$",
        va="center",
        ha="center",
        size="small",
        transform=ax1.transAxes,
    )
    if gaussian:
        summary = rf"$f_{{spot}}={_fspot:5.2f}$%"
    else:
        summary = (
            rf"$T_{{\mathrm{{app}},\phi}}={_teff:4.0f}$ K, "
            rf"$f_{{spot}}={_fspot:5.1f}$%"
            rf" ($f_{{cool}}={_fcool:5.1f}$%, $f_{{hot}}={_fhot:5.1f}$%)"
        )
    ax1.text(
        0.5,
        -0.13,
        summary,
        va="center",
        ha="center",
        size=6.0,
        transform=ax1.transAxes,
    )

    _wave = plot_data["wavelength"]
    ax2.plot(_wave, frame_data["spectrum"], "-k", lw=1)
    if gaussian:
        ax2.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax2.set_ylim(*plot_data["spectrum_ylim"])
    ax2.set_xlim(_wave[0], _wave[-1])
    ax2.set_xlabel(r"Wavelength [$\AA$]", size="small")
    ax2.set_ylabel(f"Normalized Flux", size="small")

    diagnostic_axes = _plot_phase_diagnostics(
        ax3,
        None,
        None,
        phases,
        plot_data["rv"],
        plot_data["ewr"],
        j,
        gaussian,
        photometry_for_plot,
        plot_data=plot_data["diagnostics"],
    )

    ax2.tick_params(
        axis="both",
        which="both",
        labelsize="small",
        right=True,
        top=True,
        direction="in",
        width=0.7,
    )

    for axes in [ax2, *diagnostic_axes]:
        axes.xaxis.set_minor_locator(AutoMinorLocator(5))
        axes.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.axis("off")

    for offset_x, offset_y, text in plot_data["stellar_annotation"]:
        ax1.text(
            0.03, -0.20, text, family="monospace", size=MODEL_INFO_FONT_SIZE,
            transform=ax1.transAxes + ScaledTranslation(
                offset_x / 72.0, offset_y / 72.0, fig.dpi_scale_trans),
            ha="left", va="top",
        )
    for offset_x, offset_y, text in plot_data["region_annotation"]:
        ax1.text(
            0.03, -0.34, text, family="monospace", size=MODEL_INFO_FONT_SIZE,
            transform=ax1.transAxes + ScaledTranslation(
                offset_x / 72.0, offset_y / 72.0, fig.dpi_scale_trans),
            ha="left", va="top",
        )

    # Draw the ID separately so its position is independent of the statistics.
    if plot_data["model_id_text"]:
        ax1.text(
            0.03, -0.49, plot_data["model_id_text"],
            family="monospace", size=MODEL_INFO_FONT_SIZE,
            transform=ax1.transAxes, ha="left", va="top",
        )

    target = (
        Path(output_path)
        if output_path is not None
        else Path(frame_directory) / f"frame_{j:05d}.png"
    )
    fig.savefig(target, format=target.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def _save_gif(frame_paths, gif_path):
    """Encode indexed PNG frames with Pillow; replace the GIF only after validation."""
    gif_path = Path(gif_path)
    frames = []
    temporary_gif = None
    try:
        for path in frame_paths:
            with Image.open(path) as image:
                # Palette frames reduce RAM use; GIF supports at most 256 colors.
                frames.append(
                    image.convert("RGB").convert(
                        "P", palette=Image.Palette.ADAPTIVE, colors=256
                    )
                )
        if not frames:
            raise ValueError("Cannot create a GIF without frames.")
        # Tight figure crops can differ slightly; pad rather than stretch the plots.
        size = (
            max(frame.width for frame in frames),
            max(frame.height for frame in frames),
        )
        for index, frame in enumerate(frames):
            if frame.size != size:
                canvas = Image.new("RGB", size, "white")
                canvas.paste(frame)
                frames[index] = canvas.convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=256
                )
                frame.close()
                canvas.close()
        duration = [200] * len(frames)
        duration[-1] = 3000
        with tempfile.NamedTemporaryFile(
            dir=gif_path.parent, prefix=".gif-", suffix=".gif", delete=False
        ) as stream:
            temporary_gif = Path(stream.name)
        frames[0].save(
            temporary_gif,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
            optimize=False,
        )
        # Read every encoded frame before replacing an existing result or deleting PNGs.
        with Image.open(temporary_gif) as image:
            for index in range(image.n_frames):
                image.seek(index)
                image.load()
        temporary_gif.replace(gif_path)
    finally:
        for frame in frames:
            frame.close()
        if temporary_gif is not None:
            temporary_gif.unlink(missing_ok=True)


def save_simulation_plots(
    star,
    grid_size,
    phases,
    integrated_stellar_spectrum,
    ccf_rv_shift,
    final_2d_surface_flux_sum,
    outdir,
    cpu_use,
    ohfe_ratio,
    stellar_info_text,
    region_info_text,
    band,
    spectrum_mode,
    spectrum_source,
    *,
    region_cube=None,
    region_brightness=None,
    region_fspot=None,
    photometry_for_plot=None,
    make_plots=True,
    make_gif_animation=False,
):
    """Save an all-phase PDF summary and optionally a phase animation."""
    if not make_plots and not make_gif_animation:
        return
    if len(phases) == 0:
        raise ValueError("Simulation plots require at least one phase.")
    if photometry_for_plot is not None:
        # All bands use the same phases; do not silently interpolate light curves.
        if not np.array_equal(
            np.asarray(phases, dtype=np.float32),
            np.asarray(photometry_for_plot["phases"], dtype=np.float32),
        ):
            raise ValueError("Photometry and spectral animation phases must match.")
        if any(
            np.shape(curve) != (len(phases),)
            for _, curve in photometry_for_plot["curves"]
        ):
            raise ValueError("Photometry curves must contain one value per phase.")
    output_directory = Path(outdir).parent
    output_directory.mkdir(parents=True, exist_ok=True)
    # Reuse one flux scale for this band's animation; Gaussian brightness is fixed.
    surface_flux_max = None
    if spectrum_source != "gaussian_profile":
        surface_flux_max = float(np.nanmax(final_2d_surface_flux_sum))
        if not np.isfinite(surface_flux_max) or surface_flux_max <= 0.0:
            raise ValueError(f"Invalid shared surface-flux maximum: {surface_flux_max}")
    # Workers receive only fixed plot data and the current phase's spectrum/map.
    plot_data = _prepare_band_plot_data(
        star,
        grid_size,
        integrated_stellar_spectrum,
        ccf_rv_shift,
        ohfe_ratio,
        band,
        spectrum_mode,
        spectrum_source,
        photometry_for_plot,
        phases=phases,
        stellar_info_text=stellar_info_text,
        region_info_text=region_info_text,
        region_brightness=region_brightness,
        surface_flux_max=surface_flux_max,
    )
    func = partial(_plot_prepared_frame, plot_data=plot_data)

    def frame_input(index):
        return _prepare_frame_data(
            star,
            index,
            phases,
            integrated_stellar_spectrum,
            final_2d_surface_flux_sum,
            gaussian=spectrum_source == "gaussian_profile",
            region_cube=region_cube,
            region_fspot=region_fspot,
        )

    if make_plots:
        pdf_path = output_directory / f"{Path(outdir).name}_{band}.pdf"
        logger.info("Saving phase-summary PDF for %s...", band)
        logger.debug("Phase-summary PDF: %s", pdf_path)
        # The last frame includes every configured phase in the diagnostic panel.
        func(frame_input(len(phases) - 1), output_path=pdf_path)
    if not make_gif_animation:
        return

    logger.info("Rendering animation frames for %s...", band)
    temporary_root = output_directory / "gif_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    model_id = getattr(star, "model_id", "model")
    frame_directory = Path(
        tempfile.mkdtemp(prefix=f"{band}_{model_id}_", dir=temporary_root)
    )
    func = partial(func, frame_directory=frame_directory)

    frame_paths = [
        frame_directory / f"frame_{index:05d}.png" for index in range(len(phases))
    ]
    gif_path = output_directory / f"{Path(outdir).name}_{band}.gif"
    try:
        quiet = logger.getEffectiveLevel() >= logging.WARNING
        frames = (frame_input(index) for index in range(len(phases)))
        pqdm(frames, func, n_jobs=cpu_use, total=len(phases), disable=quiet)
        logger.info("Saving GIF for %s...", band)
        logger.debug("GIF: %s", gif_path)
        _save_gif(frame_paths, gif_path)
    except Exception:
        logger.error("Animation failed; PNG frames retained in %s", frame_directory)
        raise
    if getattr(getattr(star, "output_options", None), "save_intermediate", False):
        logger.info("Debug PNG frames retained in %s", frame_directory)
    else:
        shutil.rmtree(frame_directory)
        if not any(temporary_root.iterdir()):
            temporary_root.rmdir()


@mpl.rc_context(PLOT_STYLE)
def plot_ccf_profile(
    output_path,
    *,
    phase,
    band,
    coarse_grid,
    coarse_ccf,
    coarse_peak_rv=None,
    fine_grid=None,
    fine_ccf=None,
    raw_rv=np.nan,
    failure_reason="",
):
    """Show measured CCFs and the template-relative peak before mean subtraction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    try:
        coarse_axis, fine_axis = axes
        coarse_axis.plot(coarse_grid, coarse_ccf, color="black", lw=1)
        coarse_axis.set_title("Coarse search", fontsize="small")
        if coarse_peak_rv is not None:
            coarse_axis.axvline(
                coarse_peak_rv, color="tab:grey", ls="--", label="Selected coarse peak"
            )
            coarse_axis.legend(fontsize="x-small")
        if fine_grid is not None:
            fine_axis.plot(fine_grid, fine_ccf, "-o", color="black", lw=1, ms=2)
            if np.isfinite(raw_rv):
                fine_axis.axvline(
                    raw_rv,
                    color="tab:red",
                    ls="--",
                    label=f"Refined RV: {raw_rv:.6f} km/s",
                )
                fine_axis.legend(fontsize="x-small")
        else:
            fine_axis.text(
                0.5,
                0.5,
                "Fine search not performed",
                ha="center",
                va="center",
                transform=fine_axis.transAxes,
            )
        fine_axis.set_title("Fine search", fontsize="small")
        for axis in axes:
            axis.set_xlabel("Template-relative RV [km/s]", fontsize="small")
            axis.set_ylabel("CCF (mean-subtracted dot product)", fontsize="small")
            axis.tick_params(labelsize="small")
        status = f"; failed: {failure_reason}" if failure_reason else ""
        fig.suptitle(
            f"{band}; phase {phase:.6f}{status}\nRV before phase-mean subtraction",
            fontsize="small",
        )
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)


@mpl.rc_context(PLOT_STYLE)
def plot_cc(drvs, cc, new_rv, phase, band, properties, ccf_plot_dir):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(drvs, cc, "k-", lw=0.7)

    if new_rv is not None:
        ax.axvline(new_rv, ls="--", lw=1, alpha=0.5, c="tab:red")

        ax.text(
            0.05,
            0.95,
            f"RV: {new_rv:1.4f} km/s",
            transform=ax.transAxes,
            size="x-small",
        )
    ax.text(
        0.05,
        0.85,
        f'Prominences: {properties["prominences"]}',
        transform=ax.transAxes,
        size="x-small",
    )
    ax.text(
        0.05,
        0.75,
        f'left_bases: {properties["left_bases"]}',
        transform=ax.transAxes,
        size="x-small",
    )
    ax.text(
        0.05,
        0.65,
        f'right_bases: {properties["right_bases"]}',
        transform=ax.transAxes,
        size="x-small",
    )

    ax.set_xlabel(r"Velocity [km/s]", size="small")
    ax.set_ylabel("CCF", size="small")
    ax.tick_params(
        axis="both",
        which="both",
        labelsize="small",
        right=True,
        top=True,
        direction="in",
        width=0.7,
    )

    out_dir_pre = f'./outputs/{ccf_plot_dir.split("/")[2]}'
    os.makedirs(f"{out_dir_pre}/CCF", exist_ok=True)
    fig.savefig(
        f"{out_dir_pre}/CCF/{band}_phase_{phase:1.3f}.png",
        format="png",
        bbox_inches="tight",
    )
    plt.close(fig)


@mpl.rc_context(PLOT_STYLE)
def save_surface_temperature_frames(
    cube,
    y,
    z,
    outdir="phase_maps",
    sentinel_offdisk=0,
    cmap="inferno",
    add_colorbar=True,
    phase_values=None,
    black_spot_sentinel=None,
):
    """Save phase maps with a shared Kelvin scale; black markers are not temperatures."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.debug("Surface temperature maps: %s", outdir.absolute())
    Y, Z = np.meshgrid(y, z, indexing="ij")  # axis0=y, axis1=z
    disk_mask = (Y * Y + Z * Z) <= 1.0

    values = cube[:, disk_mask]
    physical = np.isfinite(values) & (values != sentinel_offdisk)
    if black_spot_sentinel is not None:
        physical &= values != black_spot_sentinel
    values = values[physical]
    # Include small regions and use the same temperature limits for every phase.
    vmin, vmax = (
        (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
    )
    uniform_temperature = values.size > 0 and vmin == vmax
    if uniform_temperature:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    temperature_cmap = mpl.colormaps.get_cmap(cmap).copy()
    temperature_cmap.set_bad("white")

    for phase_index, frame in enumerate(cube):
        black_mask = np.zeros(frame.shape, dtype=bool)
        if black_spot_sentinel is not None:
            black_mask = disk_mask & (frame == black_spot_sentinel)
        masked = (
            ~disk_mask | (frame == sentinel_offdisk) | ~np.isfinite(frame) | black_mask
        )
        # Transpose so horizontal axis is y and vertical axis is z.
        frame_plot = np.ma.array(frame.T, mask=masked.T)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
        try:
            im = ax.imshow(
                frame_plot,
                origin="lower",
                extent=(-1, 1, -1, 1),
                interpolation="nearest",
                cmap=temperature_cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            if black_mask.any():
                # Overlay black pixels; off-disk pixels remain white.
                overlay = np.zeros((*frame.T.shape, 4))
                overlay[black_mask.T, 3] = 1.0
                ax.imshow(
                    overlay,
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    interpolation="nearest",
                )
                ax.legend(
                    handles=[Patch(facecolor="black", label="Zero-emission region")],
                    loc="upper right",
                    fontsize=7,
                )
            ax.set_xlabel(r"$y/R_\star$")
            ax.set_ylabel(r"$z/R_\star$")
            title = f"Surface temperature\nFrame {phase_index:05d}"
            if phase_values is not None:
                title += rf" ($\phi={phase_values[phase_index]:.3f}$)"
            ax.set_title(title, fontsize=10)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            if add_colorbar and values.size:
                cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
                cbar.set_label("Surface temperature [K]")
                if uniform_temperature:
                    cbar.set_ticks([float(values[0])])
            fig.savefig(outdir / f"phase_{phase_index:05d}.png")
        finally:
            plt.close(fig)
