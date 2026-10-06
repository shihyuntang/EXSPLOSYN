# Phase lists

These files specify simulation sampling phases, one numeric value per line.
They retain the values from the former project-root files, including negative
phases and different sampling ranges. They are not interchangeable defaults.

Select a file in the simulation configuration:

```ini
[phase_sampling]
phase_file : configs/phases/phases_obs.txt
```

Run from the project root: phase_file is resolved relative to the working directory,
not the configuration directory. Absolute paths are also accepted. A blank or
None value uses phase_step instead. Bolometric diagnostics use these same phases;
they do not override the configured sampling.

Historical configurations have updated phase-file paths, but their other schemas
and fitting interfaces still require migration before use.
