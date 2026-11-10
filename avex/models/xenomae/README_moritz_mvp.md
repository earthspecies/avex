# XC_1M embeddings

Copy `xenomae.py` and `model.safetensors` into your codebase. Install the three
dependencies in `requirements.txt` (Python 3.12 or 3.13). No Hugging Face runtime
or research repo installation is required.

```python
import torch
from xenomae import load_model, preprocess_audio

model = load_model()  # defaults to model.safetensors beside xenomae.py

# Supply audio from your existing loader: (samples,) or (channels, samples).
# Use floating point samples and pass their actual sample rate.
spectrogram = preprocess_audio(waveform, sample_rate=44100)
with torch.inference_mode():
    embedding = model(spectrogram)  # float32 tensor, shape (1, 768)
    # Or: tokens = model(spectrogram, return_tokens=True)  # (1, 257, 768)

array = embedding.numpy()  # use in your downstream code
```

For CUDA, use `load_model(device="cuda")` and `model(spectrogram.to("cuda"))`;
convert outputs with `embedding.cpu().numpy()`. Keep inference in float32.

Preprocessing averages channels to mono, resamples to 32 kHz, peak-normalizes,
and pads/truncates to five seconds. It then computes a power mel spectrogram
(128 bins, FFT/window 1024, hop 320, minimum 50 Hz), converts to dB with an 80 dB
floor, applies `2 * (dB + 80) / 80 - 1`, and pads frames to `(1, 1, 128, 512)`.
Silence remains finite. Channel-first input matches torchaudio; transpose a
SoundFile `(samples, channels)` array before passing it in.

Split longer recordings into five-second clips before preprocessing. Batch
them with `torch.cat([preprocess_audio(clip, sample_rate) for clip in clips])`.
The preprocessing matches the released HF feature extractor for a single clip.

For cosine search, fit a mean on a representative collection of raw embeddings,
then reuse that fixed mean for both indexed embeddings and queries:

```python
import torch.nn.functional as F

reference_mean = reference_embeddings.mean(dim=0, keepdim=True)
indexed = F.normalize(reference_embeddings - reference_mean, dim=-1)
query = F.normalize(embedding - reference_mean, dim=-1)
scores = query @ indexed.T
```

Raw MAE embeddings have strongly similar directions, so centering helps cosine
search. Search/clustering quality still needs checking on your intended data.

The safetensors file contains the encoder weights from `XC_1M.ckpt`, with only
the `encoder.` key prefix removed. Numerical parity is checked against the
original encoder on preprocessed audio before the weights are supplied.
The weights are generated and ignored by git; distribute them alongside the
Python file. Code is MIT. The parent repo declares weights CC BY-NC-SA 4.0.
