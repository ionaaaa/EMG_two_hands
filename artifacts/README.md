# Artifact layout

Raw models, reports, and logs are not committed to ordinary Git. This directory
keeps their organization rules and the versioned index
`artifact_manifest.csv`; the generated subdirectories are ignored.

```text
artifacts/
├── models/<model>/<split>/<mode>/<run-id>/
├── reports/
├── logs/
├── duplicate-review/
└── legacy-unclassified/
```

New training runs use this identifier when no explicit `--output-dir` is
provided:

```text
{model}__{split}__{mode}__{timestamp}__seed-{seed}
```

For example,
`effie__cross-session-cv__freeze-backbone__2026-08-27T14-30-00__seed-42`.
The current default artifact root remains `apps/desktop` until the indexed
historical artifacts are reviewed and migrated. Pass `--artifacts-root
artifacts` to direct a new run to this layout.

`artifact_manifest.csv` is a read-only migration index: `new_path` is a
proposed destination, not evidence that a historical file has moved. Its date
field is derived from filesystem modification time and requires review before a
formal historical rename. Files marked `suspected-duplicate` or
`temporary-review` must be retained until an owner confirms the disposition.
