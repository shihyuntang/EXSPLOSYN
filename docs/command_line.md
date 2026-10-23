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

### `--output-dir PATH`

Save results directly into this folder, replacing `output_root` and `run_name`.
Nothing is added to the folder name, even with `--paths`.
Relative paths start from your terminal's current folder; full paths choose
that location directly. The configuration file's location does not affect this.

You can also use `~` or `$HOME` for your home folder. Both single and double
quotes work, for example `'~/results/trial_02'` or `"~/results/trial_02"`.
Put quotes around any path containing spaces.

```bash
python main.py --config demo --output-dir outputs/trial_02
python main.py --config /path/to/model.cfg --output-dir '~/results/trial_02'
```

Without this option, results go to `output_root/run_name`; `output_root`
defaults to `outputs`. See {ref}`output settings <output>`.


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
