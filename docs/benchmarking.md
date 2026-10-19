# Performance benchmark

> **Help us benchmark EXSPLOSYN!**
> EXSPLOSYN is still in early development, and your results will help us
> understand how it performs across different computers.

After setting up and running the demos, activate the EXSPLOSYN environment
and run from the source repository:

```sh
python benchmark.py
```

The required dependencies are included in `environment.yml`. No separate
installation or benchmark configuration is needed.

The script reuses the `[paths]` settings in `configs/demo.cfg` for photometry/PHX and
`configs/demo_lkca4.cfg` for FeOH.

Missing PHOENIX inputs may be downloaded
during preparation; FeOH is skipped if its library is unavailable.

## What it runs

The fixed workload uses **five active regions, 100 phases, a 200 × 200 disk
grid**, comparing **one process against four processes**. It tests:

- One TESS light curve.
- One PHOENIX RV band.
- SYNMAST FeOH RVs and equivalent-width ratios, when available.

Each case performs **two warm-ups and five timed evaluations**, then makes one
separate memory measurement. Every evaluation rebuilds the temperature-dependent
local spectra, including interpolation, limb darkening and spectral preprocessing,
before calculating the model. Parameters stay fixed between evaluations so every
computer performs the same workload.

Configuration loading, initial downloads and the **fixed 3800 K RV CCF template**
are prepared once, outside the evaluation timings. Current source readers may
read cached library files during local preparation; these reads are included.
The model runs in fitting mode, without plots or simulation exports. Existing
NumPy/Numba thread settings are preserved.

## What you get

The terminal shows the computer/CPU, OS, core and worker counts, individual
evaluation times, the **average and median time**, and the **speedup relative
to one process** for each case. The four-process test is skipped if fewer than
four CPUs are available. Reports
are saved in a new timestamped folder under `benchmark_results/`:

- `benchmark_summary.csv`: hardware and performance statistics per case, including
  total time and separate averages for local preparation and model calculation.
- `benchmark_details.json`: versions, effective settings, individual timings,
  thread information, warnings and output checks.

To share your results, send the **entire timestamped results folder** to the
developers, including both files above. You can zip the folder before sending it.
Nothing is uploaded automatically. Reports omit hostnames, serial numbers and configured library
paths. Memory values are sampled process-tree RSS sums, not unique physical
RAM; unavailable or incomplete readings are marked in the detailed report.

Run while the computer is otherwise idle; connect laptops to power. Windows
has not yet been validated. For optional settings, use `python benchmark.py -h`.
