# EXSPLOSYN roadmap

Updated: 2026-10-13. `[x]` completed; `[ ]` pending. Add new ideas to the backlog.

## v0.9.1 — Simulation testing baseline

- [x] Run basic simulations with RV, EWR, and photometry.
- [x] Organize code, configs, caches, and outputs; keep development tools in `dev/`.
- [x] Provide installation files: `environment.yml`, `pyproject.toml`, and `MANIFEST.in`.
- [x] Check installation files and dependency consistency.
- [x] Save the reviewed development baseline in Git.
- [ ] Create a release tag after preparing the public repository history.
- Goal: move development to the Mac mini; complete public documentation is not required.

## v0.9.x — Documentation and group testing

- [ ] Validate installation and a basic simulation on the Mac mini.
- [ ] Provide an obtainable RASSINE version with the required patches.
- [ ] Create basic Read the Docs pages: installation, spectral data setup, and a first simulation tutorial.
- [ ] Explain the main config settings, outputs, and known limitations.
- [ ] Invite group members to test the code; record bugs and user feedback.
- Goal: let others follow the tutorial, run simulations, and help find problems.

## v1.0.0 — First official public release

- [ ] Validate installation and basic simulation on a clean environment.
- [ ] Complete basic installation instructions and simulation tutorials.
- [ ] Address blocking bugs found during group testing.
- [ ] Complete citations, licensing, and release information.
- [ ] Publish the first official release.
- Goal: basic simulation and documentation are ready for public use.

## v1.1 — Release accompanying the ApJ paper

- [ ] Fix issues affecting usability or scientific results, guided by group testing.
- [ ] Validate the paper's models, outputs, and reproducibility.
- [ ] Finalize the feature scope and publish the version associated with the paper.
- Goal: provide the validated version associated with the paper.

## Backlog / new ideas — Version not assigned

- [ ] Fix geometry for regions larger than 80%; currently rejected with an error.
- [ ] Include the vrt effect in SYNMAST intensity spectra.
- [ ] Evaluate LDTk fit starting values and band-order dependence.
- [ ] Review exact-equality assertions with differences around 1e-14 in multiprocessing tests.
- [ ] Support PHOENIX 2025.
- [ ] Support NPZ and CSV spectral libraries.
- [ ] Update fitting workflows and improve repeated model calls.
- [ ] Measure performance on the Mac mini before choosing further optimizations.
