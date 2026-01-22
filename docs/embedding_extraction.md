# Embedding Extraction and Feature Representations

## Understanding `return_features_only=True`

When loading models with `return_features_only=True`, the model returns **unpooled features** instead of classification logits. This preserves temporal and spatial information, providing richer representations for downstream tasks.

```python
# Load model for embedding extraction
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
model.eval()

# Get unpooled features
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (batch, time_steps, feature_dim)
```

## Model-Specific Output Formats

Different models return features in different formats when `return_features_only=True`:

### BEATs (Bidirectional Encoder representation from Audio Transformers)

**Output Shape**: `(batch, time_steps, 768)`

**Key Characteristics**:
- Each time step contains **8 embeddings** (one per frequency band)
- Structure: `[T0_0, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6, T0_7, T1_0, T1_1, ...]`
- **Frame rate**: 6.25 Hz (not 100 Hz)
  - Calculated as: 100 Hz / 16 (patch embedding size) = 6.25 Hz
  - For 16 kHz input audio
- Feature dimension: 768 per embedding

**Example**:
```python
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, ~31, 768)
# 31 frames ≈ 5 seconds * 6.25 Hz
# Each frame has 768-dimensional features representing 8 frequency bands
```

**Understanding BEATs Frame Structure**:
- Audio at 16 kHz: 16,000 samples per second
- Patch embedding size: 16 samples
- Base frame rate: 100 Hz (1000 ms / 10 ms per frame)
- Actual frame rate after patching: 100 Hz / 16 = **6.25 Hz**
- Each frame covers: 1 / 6.25 = **160 ms** of audio
- For 5 seconds of audio: 5 * 6.25 = **31.25 frames**

**Use Cases**:
```python
# Option 1: Pool manually for classification
pooled = features.mean(dim=1)  # (batch, 768)

# Option 2: Use specific frequency band
band_0 = features[:, :, :96]  # First frequency band (assuming 96-dim per band)

# Option 3: Use for sequence modeling
# Features preserve temporal structure for RNNs, Transformers, etc.
```

### EAT (Efficient Audio Transformer)

**Output Shape**: `(batch, num_patches, 768)`

**Key Characteristics**:
- Returns unpooled patch embeddings from transformer backbone
- Includes CLS token as first patch (index 0)
- Number of patches depends on input length and patch size
- Feature dimension: 768 per patch

**Example**:
```python
model = load_model("sl_eat_animalspeak_ssl_all", return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, 513, 768)
# 513 patches = 1 CLS token + 512 spectrogram patches
```

**Use Cases**:
```python
# Option 1: Use CLS token (typically most informative)
cls_token = features[:, 0]  # (batch, 768)

# Option 2: Mean pooling over all patches
pooled = features.mean(dim=1)  # (batch, 768)

# Option 3: Exclude CLS token and pool
spatial_features = features[:, 1:]  # Exclude CLS token
pooled = spatial_features.mean(dim=1)  # (batch, 768)
```

### EfficientNet

**Output Shape**: `(batch, channels, height, width)`

**Key Characteristics**:
- Returns spatial feature maps before global average pooling
- Preserves 2D spatial structure of spectrogram
- Channel and spatial dimensions depend on model variant

**Example**:
```python
model = load_model("efficientnet_animalspeak", return_features_only=True, device="cpu")
audio = torch.randn(1, 16000 * 5)  # 5 seconds at 16kHz
features = model(audio, padding_mask=None)
# features.shape = (1, 1280, 4, 5) for EfficientNet-B0
# 1280 channels, 4x5 spatial dimensions
```

**Use Cases**:
```python
# Option 1: Global average pooling
pooled = features.mean(dim=[2, 3])  # (batch, 1280)

# Option 2: Max pooling
pooled = features.amax(dim=[2, 3])  # (batch, 1280)

# Option 3: Flatten for spatial awareness
flattened = features.flatten(1)  # (batch, 1280*4*5)
```

See `examples/05_embedding_extraction.py` for comprehensive examples of embedding extraction with different models.
