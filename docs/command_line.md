# Command-line options

Command-line options select a configuration, run mode, and optional overrides.
For configuration-file settings, see the [configuration manual](configuration.md).

From the repository root, use `python main.py [options]`. If EXSPLOSYN is
installed as a package, `exsplosyn [options]` and
`python -m exsplosyn [options]` accept the same flags.

## Usual options

### `--config FILE`

Select the simulation configuration.

- In a source checkout, a bare filename is resolved inside `configs/`.
- `.cfg` is added when no extension is given.
- Explicit relative paths use the working directory; absolute paths are accepted.
- The source-checkout default is `demo.cfg`. An installed package requires an
  explicit configuration path.

### `-o SUFFIX` / `--output-suffix SUFFIX`

Append a suffix to `[output] save_folder_name` for this run, before any named
path-profile suffix. For example, `-o test` with `save_folder_name : demo`
saves results in `outputs/demo_test/` when using the base `[paths]` settings.

`-out` is also accepted as an alias.

### `--quiet`

Suppress EXSPLOSYN informational messages and progress bars. Warnings and
errors remain visible. This flag does not change which files are saved.

`-silence` is also accepted as an alias. Third-party download tools may still
show their own messages or progress indicators.

### `-h` / `--help`

Show the command-line help and exit.

### `--version`

Show the EXSPLOSYN version and exit.

## Run modes

### `--mode simulation`

The default mode. Save CSV results and follow the `[output]` settings for
summary and continuum-fitting plots, GIF animations, and final spectral exports.

### `--mode fitting`

Keep results in RAM and disable result-file exports and plots, regardless of
the `[output]` settings. EXSPLOSYN informational messages and progress bars
are also suppressed; warnings and errors remain visible.

For Python callers, the CLI function returns the in-memory results. Running
this mode only as a terminal command does not save those results for later use.
First-time spectral downloads and reusable input-cache preparation can still
require disk access; third-party tools may display their own messages.

### `--mode debug`

Enable all existing outputs, regardless of the `[output]` switches. These
include final results, intermediate spectral exports, surface-temperature
maps, continuum-fitting plots, and CCF-profile diagnostics.

Debug logging also identifies source spectra, cache use, and interpolation
parameters and grid points. `--quiet` suppresses these informational and debug
messages while retaining the debug output files.

## Advanced options

### `--paths NAME`

Use `[paths:NAME]` to override entries in the base `[paths]` section.
This is useful when running the simulation on different machines.
Unlisted entries are inherited from `[paths]`. The profile must be defined in
the selected configuration file.

Without this flag, EXSPLOSYN uses `[paths]` only. Explicitly selecting a named
profile appends its name to the results folder; using the base paths adds no
profile suffix.

See {ref}`named path profiles <named-path-profiles>` for configuration
details and the example at the end of `demo.cfg`.
