# Examples

The `examples/` directory contains comprehensive examples demonstrating various usage patterns:

| Example | Description |
|---------|-------------|
| `00_quick_start.py` | Basic model loading and testing |
| `01_basic_model_loading.py` | Loading pre-trained models with checkpoints and class mappings |
| `02_checkpoint_loading.py` | Working with default and custom checkpoints from YAML configs |
| `03_custom_model_registration.py` | Creating and registering custom model classes and ModelSpecs |
| `04_training_and_evaluation.py` | Full training loop and evaluation examples |
| `05_embedding_extraction.py` | Feature extraction with `return_features_only=True` (unpooled features) |
| `06_classifier_head_loading.py` | Understanding classifier head behavior with different `num_classes` settings |
| `colab_sl_beats_demo.ipynb` | Google Colab demo for the sl-beats model |

## Package Structure

```
avex/
├── __init__.py              # Main API exports and version
├── api/                     # Public API layer
│   ├── configs/            # Official model configurations
│   │   └── official_models/  # YAML configs for official models
│   └── list_models.py      # CLI utility for listing models
├── cli.py                   # Command-line interface
├── configs.py               # Pydantic configuration models
├── data/                    # Data loading and processing
├── evaluation/              # Evaluation utilities
├── metrics/                 # Evaluation metrics
├── models/                  # Model implementations
│   ├── utils/              # Model utilities (factory, load, registry)
│   ├── probes/             # Probe implementations
│   ├── beats/              # BEATs model components
│   ├── eat/                # EAT model components
│   └── atst_frame/         # ATST-Frame model components
├── preprocessing/           # Audio preprocessing
├── training/                # Training utilities
├── utils/                   # Utility functions
├── run_train.py            # Training entry point
└── run_evaluate.py         # Evaluation entry point
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unittests/
uv run pytest tests/integration/
uv run pytest tests/consistency/

# Run with coverage
uv run pytest --cov=avex
```
