# Training and Evaluation

This guide covers two approaches to training and evaluation:

1. **API-based approach**: Using the Python API directly for custom training loops
2. **Script-based approach**: Using `run_train.py` and `run_evaluate.py` for supervised learning with configuration files

## API-Based Training and Evaluation

### Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from avex import build_model

# Create a backbone for training (attach your own head or use a probe)
model = build_model("efficientnet", device="cpu")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_audio, batch_labels in train_loader:
        # Forward pass
        outputs = model(batch_audio, padding_mask=None)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save checkpoint
torch.save(model.state_dict(), "checkpoints/my_model.pt")
```

For complete training examples with data loading and evaluation, see `examples/04_training_and_evaluation.py`.

### Evaluation

```python
import torch
from avex import load_model

# Load pre-trained model with checkpoint
model = load_model("esp_aves2_sl_beats_all", device="cpu")
model.eval()

# Run inference
with torch.no_grad():
    audio = torch.randn(1, 16000 * 5)  # 5 seconds of audio
    outputs = model(audio, padding_mask=None)
    predictions = torch.softmax(outputs, dim=-1)

    # If model has label mapping, get human-readable labels
    if hasattr(model, "label_mapping"):
        top_k = 5
        probs, indices = torch.topk(predictions, top_k)
        for prob, idx in zip(probs[0], indices[0]):
            label = model.label_mapping["index_to_label"][idx.item()]
            print(f"{label}: {prob.item():.4f}")
```

For complete evaluation examples, see `examples/04_training_and_evaluation.py`.

## Script-Based Training and Evaluation

For supervised learning experiments, the framework provides dedicated scripts that handle the full training and evaluation pipeline using YAML configuration files.

> **Note**: Both `run_train.py` and `run_evaluate.py` require the `[dev]` dependencies. Install them with:
> ```bash
> # With uv
> uv sync --group project-dev
> # Or
> uv add "avex[dev]"
>
> # With pip
> pip install -e ".[dev]" --extra-index-url https://oauth2accesstoken@us-central1-python.pkg.dev/okapi-274503/esp-pypi/simple/
> ```
>
> The `[dev]` extra includes dependencies like `pytorch-lightning`, `mlflow`, `wandb`, `esp-data`, `esp-sweep`, and other training/evaluation tools.

### Training with `run_train.py`

The [`run_train.py`](../avex/run_train.py) script provides a complete training pipeline with support for:

- Distributed training
- Experiment tracking (MLflow, WandB)
- Checkpoint management
- Configurable optimizers and schedulers
- Data loading with `esp-data`

**Prerequisites:**

- Install with `[dev]` dependencies (see note above)
- GCP authentication for accessing datasets and model checkpoints

**Usage:**

```bash
# Using the CLI
uv run avex train --config configs/run_configs/my_training_config.yml

# Or directly with Python
uv run python avex/run_train.py configs/run_configs/my_training_config.yml

# With config patches (override config values)
uv run avex train --config configs/run_configs/my_training_config.yml --patch "training.lr=0.001" --patch "model.device=cuda"
```

The script expects a YAML configuration file that defines:
- Model specification (`ModelSpec`)
- Training parameters (optimizer, learning rate, epochs, etc.)
- Dataset configuration
- Experiment tracking settings
- Output directories

### Evaluation with `run_evaluate.py`

The [`run_evaluate.py`](../avex/run_evaluate.py) script provides comprehensive evaluation capabilities for:

- **Linear probing**: Training linear classifiers on frozen backbones
- **Fine-tuning**: End-to-end training of backbones with probes
- **Retrieval evaluation**: Computing retrieval metrics
- **Clustering evaluation**: Computing clustering metrics
- **Offline/Online training**: Support for both embedding pre-computation and end-to-end training

**Key features:**
- **No duplicate forward-pass**: Train, validation, and test embeddings are computed once and reused
- **End-to-end evaluation**: Linear-probe test accuracy is measured on raw audio to reflect real inference cost
- **Multiple probe types**: Supports linear, MLP, LSTM, attention, and transformer probes

**Prerequisites:**

- Install with `[dev]` dependencies (see note above)
- GCP authentication for accessing datasets and model checkpoints

**Usage:**

```bash
# Using the CLI
uv run avex evaluate --config configs/evaluation_configs/my_evaluation_config.yml

# Or directly with Python
uv run python avex/run_evaluate.py configs/evaluation_configs/my_evaluation_config.yml

# With config patches
uv run avex evaluate --config configs/evaluation_configs/my_evaluation_config.yml --patch "experiments[0].model_spec.device=cuda"
```

The evaluation script expects a YAML configuration file (`EvaluateConfig`) that defines:
- List of experiments to run (each with a `ModelSpec`)
- Dataset configuration for evaluation sets
- Training parameters for fine-tuning/probing
- Evaluation modes (`probe`, `retrieval`, `clustering`)
- Output directories for results

**Example evaluation workflow:**

1. **Offline mode**: Extract embeddings once, then train probes on pre-computed embeddings
2. **Online mode**: Train probes end-to-end with the backbone (backbone can be frozen or fine-tuned)
3. **Multiple evaluation sets**: Evaluate on multiple benchmark datasets in a single run
4. **Comprehensive metrics**: Automatically computes classification, retrieval, and clustering metrics

See the `configs/` directory for example configuration files, and check the job scripts in `jobs/` for real-world usage examples.
