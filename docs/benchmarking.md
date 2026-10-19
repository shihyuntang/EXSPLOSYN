# Performance benchmark

> **Help us benchmark EXSPLOSYN!**
> EXSPLOSYN is still in early development, and your results will help us
> understand how it performs across different computers.

After setting up and running the demos, activate the EXSPLOSYN environment
and run from the source repository:

```sh
python benchmark.py
```

The required dependencies are included in `environment.yml`. Set the `[paths]`
entries in **`configs/performance.cfg`** to your local libraries, as in the demos.
The small FeOH demo library is sufficient; if unavailable, FeOH is omitted.

## What it runs

The script uses one fixed configuration: **five active regions, 200 phases,
and a 200 × 200 disk grid**, with **TESS/TESS.Red photometry and FeOH
RV/EWR** running together. It changes only **`cpu : 1` versus `cpu : 4`**.

For each process count, it runs **two warm-ups followed by five timed runs**.
Each run calls the normal EXSPLOSYN CLI with `--quiet --mode fitting`, including
configuration reading, spectral/template preparation and model calculation.
No benchmark-specific model caches or simulation-code shortcuts are used.
Fitting mode returns results in RAM without simulation exports or plots.
NumPy/Numba thread settings are left unchanged. Initial downloads and imports
may occur during warm-up and are excluded from the reported timed runs.

## What you get

The terminal shows the CPU, OS name/version, RAM, Python version, workload,
core and worker counts, individual
evaluation times, the **average and median time**, and the **speedup relative
to one process** for the complete simulation. The four-process test is skipped
if fewer than four CPUs are available. Reports are saved in a new timestamped folder under `benchmark_results/`:

- `benchmark_summary.csv`: total-run timing statistics for one and four processes.
- `benchmark_details.json`: versions, effective settings, individual timings,
  thread information and any failure messages.

To share your results, send the **entire timestamped results folder** to the
developers, including both files above. You can zip the folder before sending it.
Nothing is uploaded automatically. Reports omit hostnames, serial numbers and
configured library paths. Hardware information is included in the JSON report.

Run while the computer is otherwise idle; connect laptops to power. Windows
has not yet been validated. For optional settings, use `python benchmark.py -h`.
