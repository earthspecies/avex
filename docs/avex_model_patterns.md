# AVEX Model Implementation Patterns

This note captures the conventions used by the existing AVEX model wrappers and
their tests. Use it as a review checklist for new backbone, checkpoint, and probe
integrations.

## Model Wrappers

- Model files expose a `Model` class, or register an explicit model-name mapping
  when the import path does not map cleanly to the public model name.
- Model wrappers inherit `ModelBase` and call `super().__init__(device=device,
  audio_config=audio_config)` unless they deliberately bypass a parent wrapper.
- Constructors accept the common AVEX surface: `num_classes`, `pretrained`,
  `device`, `audio_config`, and, when supported, `return_features_only`.
- If `num_classes is None`, backbone-style models should prefer embedding mode
  over constructing a placeholder classifier. Existing wrappers log this and set
  `return_features_only=True` where the architecture supports it.
- Model-specific options are represented in `ModelSpec` and passed through
  `_add_model_spec_params`; avoid hidden top-level YAML keys unless the loader
  also knows how to preserve them.
- Runtime-only inputs such as `num_classes` extracted from a checkpoint and
  `return_features_only` belong in loader/factory kwargs, not static YAML.

## Audio Processing

- `process_audio` is the wrapper boundary for model-specific input preparation.
- Raw-waveform wrappers either use `ModelBase.process_audio` for configured
  `AudioProcessor` behavior or clearly own their full preprocessing stack.
- The returned tensor should be on the model device and should preserve training
  viability on that device. Avoid unconditional CPU preprocessing inside hot
  training paths unless the model or dependency requires it.
- `padding_mask` is accepted by `forward` for API compatibility. If unsupported,
  the model should ignore it explicitly and tests should cover parity.

## Hook And Embedding API

- `_discover_linear_layers` is the public hook-discovery hook, even when the
  selected layers are not strictly linear layers.
- `target_layers=["all"]` and `target_layers=["last_layer"]` rely on
  `_layer_names` and `ModelBase.register_hooks_for_layers`.
- `extract_embeddings` validates that hooks are registered, clears stale hook
  outputs, runs a forward pass, returns either a single tensor or a list for
  `aggregation="none"`, and clears captured outputs afterward.
- Supported aggregation names are `mean`, `max`, `cls_token`, and `none`.
  Tensor layout handling must be documented because CNN and transformer
  features use different spatial or temporal axes.

## Checkpoint And Registry Integration

- Packaged, user-loadable model configs live under `avex/api/configs`.
- Official registry lookup currently reads model specs from
  `official_models/*.yml` and default checkpoint paths from top-level
  `checkpoint_path`.
- Checkpoint format adaptation belongs near the model wrapper when it is
  architecture-specific, for example remapping Lightning prefixes in
  `load_state_dict`.
- If a model needs a directory, multiple files, or nonstandard checkpoint
  metadata, the loader/factory/schema path should expose that need explicitly.

## Tests

- Unit tests should have a no-network path using `pretrained=False` or small
  synthetic checkpoints.
- Tests cover constructor invariants, forward shape and dtype, padding-mask
  compatibility, layer discovery, hook-based embedding extraction, config
  passthrough, and checkpoint key remapping when applicable.
- Network-backed tests should be clearly isolated or skipped unless credentials
  and cache availability are guaranteed in CI.
- For feature or training changes, add tests around the training path, not just
  standalone forward calls.

## Review Checklist

- Does the wrapper follow the common constructor and factory patterns?
- Are all model-specific config fields represented in `ModelSpec` or a documented
  loader extension point?
- Can `load_model(path_to_yaml)` and `build_model_from_spec` both instantiate the
  model users are told to configure?
- Does `pretrained=False` work offline for tests and training scaffolding?
- Are classifier, backbone, and embedding-only modes clearly separated?
- Does freezing affect only the intended backbone parameters?
- Are hook layer names stable and covered by tests?
- Are checkpoint formats validated with useful errors and tests?
