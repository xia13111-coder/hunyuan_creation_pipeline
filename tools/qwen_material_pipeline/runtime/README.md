# Local runtime data

This directory is reserved for machine-local material-pipeline dependencies:

```text
runtime/
├── models/       model weights and local material assets
├── cache/        rebuildable indexes and caches
└── projects/     private sealed replay projects
```

These directories are ignored by Git and are not part of a source release.
Configure their actual paths through the repository-root `.env`; read-only
external mounts are also supported.

Reference photographs, logs, analysis artifacts, renders, and USD results do
not belong here. Store every run under the repository `outputs/<run-id>/`.
