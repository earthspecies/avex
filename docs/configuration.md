# Configuration

## ModelSpec Parameters

The `ModelSpec` class supports various parameters for different model types:

```python
from representation_learning.configs import ModelSpec, AudioConfig

# Basic configuration
model_spec = ModelSpec(
    name="efficientnet",
    pretrained=False,
    device="cuda",
    audio_config=AudioConfig(
        sample_rate=16000,
        representation="mel_spectrogram",
        n_mels=128
    )
)

# Model-specific parameters
model_spec = ModelSpec(
    name="beats",
    use_naturelm=True,
    fine_tuned=False,
    disable_layerdrop=False
)

# CLIP-specific parameters
model_spec = ModelSpec(
    name="clip",
    text_model_name="roberta-base",
    projection_dim=512,
    temperature=0.07
)

# EAT-specific parameters
model_spec = ModelSpec(
    name="eat_hf",
    model_id="worstchan/EAT-base_epoch30_pretrain",
    fairseq_weights_path="/path/to/weights.pt",
    eat_norm_mean=-4.268,
    eat_norm_std=4.569
)
```

## Audio Requirements

**Sample Rate**: Each model expects audio at a specific sample rate (defined in its `model_spec`).

**Finding the expected sample rate:**

```python
from representation_learning import describe_model, get_model_spec

# Option 1: Use describe_model() for a formatted overview
describe_model("beats_naturelm", verbose=True)
# Prints: 🎵 Sample Rate: 16000 Hz

# Option 2: Access programmatically via get_model_spec()
spec = get_model_spec("beats_naturelm")
target_sr = spec.audio_config.sample_rate  # 16000
```

**Resampling audio (using librosa):**

For full reproducibility, use `librosa.resample` with `res_type="kaiser_best", scale=True`.

```python
import librosa
import torch
from representation_learning import get_model_spec, load_model

# Get the model's expected sample rate
spec = get_model_spec("beats_naturelm")
target_sr = spec.audio_config.sample_rate

# Load audio at original sample rate
audio, original_sr = librosa.load("audio.wav", sr=None)

# Resample if needed (use these exact parameters for reproducibility)
if original_sr != target_sr:
    audio = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
        scale=True,
    )

# Convert to tensor and add batch dimension
audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # Shape: (1, num_samples)

# Run inference
model = load_model("beats_naturelm", return_features_only=True, device="cpu")
with torch.no_grad():
    output = model(audio_tensor, padding_mask=None)
```

> **Note**: The models were trained with audio resampled using `res_type="kaiser_best"` and `scale=True`. Using different resampling methods may affect results.

## Audio Configuration

```python
from representation_learning.configs import AudioConfig

audio_config = AudioConfig(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    window="hann",
    n_mels=128,
    representation="mel_spectrogram",  # or "spectrogram", "raw"
    normalize=True,
    target_length_seconds=10,
    window_selection="random",  # or "center"
    center=True
)
```
