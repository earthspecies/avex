These fixtures pin a small set of BEATs model weights + config fields.

Each JSON fixture contains:
- config: a dict of selected `BEATsConfig` fields (the pinned subset)
- tensors: a dict mapping state_dict key -> sha256(raw_tensor_bytes)

Test:
- `tests/integration/test_official_models_regression.py`

Regenerate:
- Load the model via `load_model(<registry_key>, return_features_only=True)`
- Hash sentinel tensors from `state_dict()`
- Dump the pinned cfg subset and hashes to JSON

