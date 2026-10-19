"""Fit PHX continua with the external RASSINE tool.

RASSINE runs with the EXSPLOSYN Python interpreter. Intermediate file layout
and cache reuse behavior follow the existing simulation workflow.
"""

from contextlib import contextmanager
from itertools import cycle
import logging
import shutil
from threading import Event, Thread
from time import perf_counter
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@contextmanager
def _rassine_progress():
    """Show elapsed time while waiting, without claiming a completion percentage."""
    label = "Fitting PHX continuum with RASSINE..."
    stream = sys.stderr
    if not stream.isatty() or not logger.isEnabledFor(logging.INFO):
        logger.info(label)
        yield
        return

    started = perf_counter()
    stopped = Event()

    def display(message):
        width = max(1, shutil.get_terminal_size().columns - 1)
        stream.write("\r\033[2K" + message[:width])
        stream.flush()

    def animate():
        for symbol in cycle("|/-\\"):
            display(f"{label} {symbol} elapsed {perf_counter() - started:.0f} s")
            if stopped.wait(0.2):
                break

    # Only the display runs in a thread; the subprocess call is unchanged.
    thread = Thread(target=animate, daemon=True)
    thread.start()
    status = "failed"
    try:
        yield
        status = "done"
    finally:
        stopped.set()
        thread.join()
        display(f"{label} {status} ({perf_counter() - started:.1f} s)")
        stream.write("\n")
        stream.flush()


def flatten_phx_with_rassine(
    rassine_path, out_path, phx_flux_file, *, keep_intermediate=False
):
    """
    Fit a PHX continuum with RASSINE and save wave, flux and continuum_cubic.

    The saved flux is not divided by the continuum here. Normalization is
    performed later using flux / continuum_cubic.

    Inputs
    ------
    out_path      : base output directory (string or Path)
    phx_flux_file : filename of the PHX spectrum HDF under local_spectra/library_interpolated/
                    (with or without .h5)
    rassine_path  : path to the RASSINE repo (directory containing Rassine.py)
    keep_intermediate : preserve generated input/output pickles for debug runs

    RASSINE uses the current Python interpreter and its installed dependencies.
    The checkout must provide Rassine.py, Rassine_functions.py, and
    Rassine_config.py with the supported three-positional-argument interface.

    Returns
    -------
    outname : str
        HDF path at local_spectra/rassine_continuum/<phx_flux_file>_rassine.h5, containing
        the input flux and fitted continuum.
    """
    out_path = Path(out_path) / "local_spectra"
    rassine_dir = out_path / "rassine_continuum"
    rassine_dir.mkdir(parents=True, exist_ok=True)

    # Remove only the final HDF suffix when building cache filenames.
    base = phx_flux_file.removesuffix(".h5")
    in_hdf = out_path / "library_interpolated" / f"{base}.h5"  # input PHX HDF
    in_pkl = rassine_dir / f"{base}.p"  # temp pickle for RASSINE
    out_hdf = rassine_dir / f"{base}_rassine.h5"  # final compact HDF
    out_pkl = rassine_dir / f"RASSINE_{base}.p"  # RASSINE output pickle

    # Reuse the fitted continuum when the cached output exists.
    if out_hdf.exists():
        logger.debug("RASSINE continuum already exists: %s", out_hdf)
        return str(out_hdf)

    # Validate the external installation before creating intermediate files.
    rassine_root = Path(rassine_path).expanduser()
    for filename in ("Rassine.py", "Rassine_functions.py", "Rassine_config.py"):
        required_file = rassine_root / filename
        if not required_file.is_file():
            raise FileNotFoundError(f"Required RASSINE file not found: {required_file}")
    rassine_py = rassine_root / "Rassine.py"

    logger.debug("RASSINE continuum output: %s", out_hdf)

    # 1) Convert the full input table to the pickle format RASSINE reads.
    try:
        input_spectrum = pd.read_hdf(in_hdf)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input PHX HDF not found: {in_hdf}") from e

    input_spectrum.to_pickle(in_pkl)

    # 2) Run RASSINE with the current Python interpreter.
    # RASSINE CLI: Rassine.py <input_pickle> <output_dir> <rassine_root_dir>
    # RASSINE joins directory strings to filenames, so keep trailing slashes.
    out_dir = str(rassine_dir).rstrip("/") + "/"
    root_dir = str(rassine_root).rstrip("/") + "/"

    cmd = [
        sys.executable,
        str(rassine_py),
        str(in_pkl),
        out_dir,  # <output_dir> with trailing '/'
        root_dir,  # <rassine_root_dir> with trailing '/'
    ]
    logger.debug("Running > %s", " ".join(cmd))

    try:
        with _rassine_progress():
            process_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if process_result.stdout:
            logger.debug("RASSINE stdout:\n%s", process_result.stdout.strip())
        if process_result.stderr:
            # Successful runs may also write diagnostic messages to stderr.
            logger.debug("RASSINE stderr:\n%s", process_result.stderr.strip())
    except subprocess.CalledProcessError as e:
        # Preserve intermediate files when RASSINE fails.
        logger.error("RASSINE failed (exit %s). STDERR:\n%s", e.returncode, e.stderr)
        raise
    except OSError as e:
        raise RuntimeError(
            f"Could not launch RASSINE with Python {sys.executable!r}: {e}"
        ) from e

    # 3) Save the input flux and fitted continuum in a compact HDF.
    try:
        rassine_output = pd.read_pickle(out_pkl)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Expected RASSINE output pickle not found: {out_pkl}"
        ) from e

    try:
        wave = rassine_output["wave"]
        flux = rassine_output["flux"]
        continuum = rassine_output["output"]["continuum_cubic"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            "RASSINE output must contain wave, flux and output.continuum_cubic."
        ) from e

    try:
        wave = wave.astype(np.float32, copy=False)
        flux = flux.astype(np.float64, copy=False)
        continuum = continuum.astype(np.float64, copy=False)
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError(
            "RASSINE wave, flux and continuum_cubic must be numeric arrays."
        ) from e

    output_spectrum = pd.DataFrame(
        {"wave": wave, "flux": flux, "continuum_cubic": continuum}
    )
    output_spectrum.to_hdf(out_hdf, key="spec", complevel=5, index=False)

    # 4) Remove intermediate pickles after success, unless debug needs them.
    for pickle_path in (() if keep_intermediate else (in_pkl, out_pkl)):
        try:
            pickle_path.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Could not remove RASSINE intermediate %s: %s", pickle_path, e)

    return str(out_hdf)
