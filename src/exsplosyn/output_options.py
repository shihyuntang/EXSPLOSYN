"""Select file outputs for simulation, fitting and debug runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputOptions:
    mode: str
    in_memory: bool
    write_csv: bool
    save_spectra: bool
    make_plots: bool
    save_intermediate: bool
    make_gif_animation: bool = False


# -----------+-----+---------------+---------------+------------------------
# Mode       | CSV | Final spectra | PDF / GIF     | Intermediate debug data
# -----------+-----+---------------+---------------+------------------------
# simulation | Yes | User setting  | User settings | No
# fitting    | No  | No            | No            | No
# debug      | Yes | Yes           | Yes           | Yes
# -----------+-----+---------------+---------------+------------------------
# PDF and GIF have independent settings in simulation mode.
def resolve_output_options(
    mode, save_spectra=False, make_plots=False, make_gif_animation=False
):
    """Fitting returns results in RAM; debug enables all existing exports."""
    if mode not in {"simulation", "fitting", "debug"}:
        raise ValueError(
            f"Unsupported mode {mode!r}; use simulation, fitting or debug."
        )
    fitting = mode == "fitting"
    debug = mode == "debug"
    return OutputOptions(
        mode=mode,
        in_memory=fitting,
        write_csv=not fitting,
        save_spectra=not fitting and (debug or bool(save_spectra)),
        make_plots=not fitting and (debug or bool(make_plots)),
        save_intermediate=debug,
        make_gif_animation=not fitting and (debug or bool(make_gif_animation)),
    )
