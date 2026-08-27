# Data layout

This directory documents locally held EMG data without placing raw captures in
ordinary Git. The `datasets/` and `recordings/` subdirectories are ignored;
the documentation and migration manifest remain versioned.

```text
data/
├── datasets/<subject_id>/<session_id>/
│   ├── metadata.json
│   ├── emg.csv
│   ├── imu.csv
│   ├── events.csv
│   └── raw_packets.bin
└── recordings/<recording_id>/
    └── the same five capture files
```

`datasets/` contains structured collection sessions used by training and
evaluation. `recordings/` contains free-form recordings. Obtain original data
from the experiment owner or the team's approved object storage; do not add
raw captures to ordinary Git. Use Git LFS only when a deliberately selected,
small versioned data artifact requires it. Put minimal, redistribution-safe
test samples under `apps/desktop/tests/fixtures/` instead of this directory.

`migration-manifest.csv` records the 2026-08-27 relocation from the legacy
desktop paths. Its hashes are calculated per file; raw data is not included in
the repository.
