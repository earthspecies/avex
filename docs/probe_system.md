# Flexible Probe System

This document describes the new flexible probe system that replaces the old rigid linear probe approach. The new system allows you to configure different types of probes with various aggregation and processing strategies.

## Overview

The new probe system provides:

- **Multiple probe types**: Linear, MLP, LSTM, Attention, and Transformer probes
- **Flexible aggregation**: Mean, max, concatenation, CLS token, or no aggregation
- **Input processing options**: Flatten, sequence, pooled, or no processing
- **Probe-specific parameters**: Hidden dimensions, attention heads, LSTM configuration, etc.
- **Training overrides**: Per-probe learning rates, batch sizes, epochs, etc.
- **Backward compatibility**: Legacy configurations still work automatically

## Probe Types

### 1. Linear Probe (`"linear"`)
Simple linear classification layer. Good for baseline performance.

```yaml
probe_config:
  probe_type: "linear"
  aggregation: "mean"
  input_processing: "pooled"
  target_layers: ["layer_12"]
```

### 2. MLP Probe (`"mlp"`)
Multi-layer perceptron with configurable hidden dimensions.

```yaml
probe_config:
  probe_type: "mlp"
  aggregation: "mean"
  input_processing: "pooled"
  target_layers: ["layer_8", "layer_12"]
  hidden_dims: [512, 256]
  dropout_rate: 0.2
  activation: "gelu"
```

### 3. LSTM Probe (`"lstm"`)
Long Short-Term Memory network for sequence modeling.

```yaml
probe_config:
  probe_type: "lstm"
  aggregation: "none"
  input_processing: "sequence"
  target_layers: ["layer_6", "layer_8", "layer_10", "layer_12"]
  lstm_hidden_size: 256
  num_layers: 2
  bidirectional: true
  max_sequence_length: 1000
```

### 4. Attention Probe (`"attention"`)
Attention mechanism for sequence modeling.

```yaml
probe_config:
  probe_type: "attention"
  aggregation: "none"
  input_processing: "sequence"
  target_layers: ["layer_6", "layer_10"]
  num_heads: 8
  attention_dim: 512
  num_layers: 2
  max_sequence_length: 800
  use_positional_encoding: true
```

### 5. Transformer Probe (`"transformer"`)
Full transformer architecture for complex sequence modeling.

```yaml
probe_config:
  probe_type: "transformer"
  aggregation: "none"
  input_processing: "sequence"
  target_layers: ["layer_4", "layer_6", "layer_8", "layer_10", "layer_12"]
  num_heads: 12
  attention_dim: 768
  num_layers: 4
  max_sequence_length: 1200
  use_positional_encoding: true
```

## Aggregation Methods

### `"mean"`
Average embeddings across layers (default for backward compatibility).

### `"max"`
Take maximum values across layers.

### `"concat"`
Concatenate embeddings from all layers (requires larger probe networks).

### `"cls_token"`
Use only the CLS token from sequence-based models.

### `"none"`
No aggregation - pass embeddings directly to sequence-based probes.

## Input Processing Methods

### `"pooled"`
Pool embeddings to fixed dimension (default for backward compatibility).

### `"sequence"`
Keep sequence structure for sequence-based probes.

### `"flatten"`
Flatten all dimensions into a single vector.

### `"none"`
No processing - use embeddings as-is.

## Configuration Examples

### Basic Linear Probe (Legacy Style)
```yaml
experiments:
  - run_name: "simple_linear"
    run_config: "configs/run_configs/example_run.yml"
    pretrained: true
    layers: "layer_12"  # Legacy field
    frozen: true        # Legacy field
```

### Advanced MLP Probe
```yaml
experiments:
  - run_name: "advanced_mlp"
    run_config: "configs/run_configs/example_run.yml"
    pretrained: true
    probe_config:
      name: "advanced_mlp"
      probe_type: "mlp"
      aggregation: "concat"
      input_processing: "pooled"
      target_layers: ["layer_6", "layer_8", "layer_10", "layer_12"]
      freeze_backbone: true
      learning_rate: 3e-4  # Override global LR
      batch_size: 4        # Override global batch size
      hidden_dims: [1024, 512, 256]
      dropout_rate: 0.15
      activation: "relu"
```

### Sequence LSTM Probe
```yaml
experiments:
  - run_name: "sequence_lstm"
    run_config: "configs/run_configs/example_run.yml"
    pretrained: true
    probe_config:
      name: "sequence_lstm"
      probe_type: "lstm"
      aggregation: "none"
      input_processing: "sequence"
      target_layers: ["layer_8", "layer_12"]
      lstm_hidden_size: 256
      num_layers: 2
      bidirectional: true
      max_sequence_length: 1000
      use_positional_encoding: false
```

## Migration from Legacy System

The new system automatically handles legacy configurations:

1. **Legacy fields still work**: `layers` and `frozen` fields are automatically converted to `probe_config`
2. **No breaking changes**: Existing configurations continue to work without modification
3. **Gradual migration**: You can update configurations one at a time

### Before (Legacy)
```yaml
experiments:
  - run_name: "old_style"
    layers: "layer_12"
    frozen: true
```

### After (New Style)
```yaml
experiments:
  - run_name: "new_style"
    probe_config:
      probe_type: "linear"
      aggregation: "mean"
      input_processing: "pooled"
      target_layers: ["layer_12"]
      freeze_backbone: true
```

## Training Parameter Overrides

Each probe can override global training parameters:

```yaml
probe_config:
  # ... other config ...
  learning_rate: 5e-4    # Override global lr
  batch_size: 4          # Override global batch_size
  train_epochs: 15       # Override global train_epochs
  optimizer: "adam"      # Override global optimizer
  weight_decay: 0.001    # Override global weight_decay
```

## Best Practices

### 1. Choose Appropriate Probe Types
- **Linear**: Baseline performance, quick experiments
- **MLP**: Better performance, moderate complexity
- **LSTM**: Sequence modeling, moderate complexity
- **Attention**: Sequence modeling, higher complexity
- **Transformer**: Complex sequence modeling, highest complexity

### 2. Layer Selection
- **Single layer**: Use `["layer_12"]` for final representations
- **Multiple layers**: Use `["layer_6", "layer_8", "layer_10", "layer_12"]` for hierarchical features
- **Early layers**: Use `["layer_1", "layer_2", "layer_3"]` for low-level features

### 3. Aggregation Strategy
- **Mean/Max**: Good for classification tasks
- **Concat**: Better for complex tasks, requires larger probe networks
- **None**: Required for sequence-based probes

### 4. Input Processing
- **Pooled**: Good for classification tasks
- **Sequence**: Required for sequence-based probes
- **Flatten**: Good for spatial features

## Validation

The system automatically validates configurations:

- Required parameters for each probe type
- Compatibility between aggregation and input processing methods
- Valid parameter ranges (positive integers, valid activation functions, etc.)
- Layer name consistency

## Error Handling

Common validation errors and solutions:

### Missing Required Parameters
```yaml
# Error: MLP probe requires hidden_dims
probe_config:
  probe_type: "mlp"
  # Missing: hidden_dims

# Solution: Add required parameters
probe_config:
  probe_type: "mlp"
  hidden_dims: [512, 256]
```

### Incompatible Configuration
```yaml
# Error: cls_token aggregation requires sequence input_processing
probe_config:
  aggregation: "cls_token"
  input_processing: "pooled"

# Solution: Use sequence input_processing
probe_config:
  aggregation: "cls_token"
  input_processing: "sequence"
```

## Performance Considerations

### Memory Usage
- **Linear/MLP**: Low memory usage
- **LSTM**: Moderate memory usage
- **Attention/Transformer**: Higher memory usage

### Training Speed
- **Linear**: Fastest training
- **MLP**: Fast training
- **LSTM**: Moderate training speed
- **Attention/Transformer**: Slower training

### Inference Speed
- **Linear**: Fastest inference
- **MLP**: Fast inference
- **LSTM**: Moderate inference speed
- **Attention/Transformer**: Slower inference

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use simpler probe types
2. **Slow Training**: Use simpler probe types or reduce hidden dimensions
3. **Poor Performance**: Try different aggregation methods or layer combinations
4. **Validation Errors**: Check parameter compatibility and required fields

### Debug Mode

Enable debug logging to see detailed configuration validation:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Extensions

The system is designed to be extensible:

- **New probe types**: Easy to add new probe architectures
- **Custom aggregations**: Support for custom aggregation functions
- **Advanced processing**: More sophisticated input processing methods
- **Hyperparameter optimization**: Integration with hyperparameter search tools
