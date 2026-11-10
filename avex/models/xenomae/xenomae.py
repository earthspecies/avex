"""Standalone XC_1M encoder and its audio preprocessing. No repo or HF imports."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT
from safetensors.torch import load_file
from torch import nn

SAMPLE_RATE = 32_000
CLIP_SAMPLES = 5 * SAMPLE_RATE
DEFAULT_WEIGHTS = Path(__file__).with_name("model.safetensors")

_MEL = AT.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    win_length=1024,
    hop_length=320,
    n_mels=128,
    f_min=50.0,
    power=2.0,
)
_TO_DB = AT.AmplitudeToDB(stype="power", top_db=80.0)


def preprocess_audio(waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Convert one clip to float32 (1, 1, 128, 512) model input on CPU.

    Accepts (samples,) mono or (channels, samples) audio. Downmixes and resamples
    to 32 kHz, peak-normalizes, pads/truncates to five seconds, then applies the
    released mel/dB scaling. Split longer recordings yourself before calling.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    waveform = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
    if waveform.ndim not in (1, 2) or waveform.numel() == 0:
        raise ValueError("expected nonempty (samples,) or (channels, samples) audio")
    if not torch.isfinite(waveform).all():
        raise ValueError("audio contains NaN or infinite samples")
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform, sample_rate, SAMPLE_RATE)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    waveform = waveform[:CLIP_SAMPLES]
    waveform = F.pad(waveform, (0, CLIP_SAMPLES - waveform.numel()))

    spectrogram = _MEL(waveform).unsqueeze(0).unsqueeze(0)
    spectrogram = 2 * (_TO_DB(spectrogram) + 80) / 80 - 1
    return F.pad(spectrogram, (0, 512 - spectrogram.shape[-1]))


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(768, 3 * 768)
        self.proj = nn.Linear(768, 768)
        self.q_norm = nn.LayerNorm(64)
        self.k_norm = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, 12, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=64**-0.5)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.attn = _Attention()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = _MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class XenoMAE(nn.Module):
    """The released ViT-B/16 encoder. Use load_model() for pretrained weights."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        # The checkpoint provides the trained model's fixed sin/cos table.
        self.register_buffer("pos_embed", torch.zeros(1, 257, 768))
        self.blocks = nn.ModuleList([_Block() for _ in range(12)])
        self.norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, spectrograms: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """Return (batch, 768) CLS embeddings, or all (batch, 257, 768) tokens.

        Inputs have shape (batch, 1, 128, 512). Keep inference and similarity
        scoring in float32; callers control device placement and batching.
        """
        if spectrograms.ndim != 4 or spectrograms.shape[1:] != (1, 128, 512):
            raise ValueError("expected spectrograms of shape (batch, 1, 128, 512)")
        x = self.patch_embed(spectrograms) + self.pos_embed[:, 1:]
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x if return_tokens else x[:, 0]


def load_model(weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu") -> XenoMAE:
    """Load the bundled safetensors encoder in float32 and evaluation mode."""
    model = XenoMAE()
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model.float().to(device).eval()
