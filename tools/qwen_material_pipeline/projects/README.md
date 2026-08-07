# Sealed material projects

This directory contains the loader package only. Asset-specific sealed projects
may include CAD topology, photograph hashes, masks, and per-part assignments, so
they are local data and are not included in the public source release.

Live inference does not require a sealed project. To maintain a private replay
project, place its manifest, planner, template, catalog, evidence, and dependency
lock in a separate subdirectory and keep that data under its own access policy.
