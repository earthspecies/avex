"""Packaged checkpoint URI mappings.

This package stores lightweight YAML files that map stable logical names to
checkpoint URIs (e.g. `gs://...`, `hf://...`). Unlike `official_models`, these
files are *not* treated as user-facing "loadable models" and therefore do not
populate the model registry.
"""
