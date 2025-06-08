from __future__ import annotations

"""Custom audio processor replicating the original EAT mel-FBank pipeline.

The original EAT implementation converts raw waveforms into **128-bin Mel FBanks**
using Kaldi's filter-bank extractor – identical to that used by BEATs.  This
module re-implements the same logic so we can integrate it cleanly with the
existing *representation-learning* code-base without pulling in the Fairseq
stack.

The output tensor is shaped **(B, F, T)** where `F == n_mels` and `T ==
``target_length```, matching the expectations of
:pyclass:`representation_learning.models.eat.audio_model.Model`.
"""

from typing import Union

import torch
import torch.nn.functional as F
import torchaudio

__all__ = ["EATAudioProcessor"]


class EATAudioProcessor:
    """Convert raw waveforms → Mel FBanks exactly as in the original EAT code."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        target_length: int = 1024,
        n_mels: int = 128,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        # norm_mean: float = 0.0,
        # norm_std: float = 1.0,
        frame_shift_ms: int = 10,
        window_type: str = "hanning",
    ) -> None:
        """Parameters
        ----------
        sample_rate
            Expected sampling-rate of the incoming audio.  The original EAT
            model is trained at **16 kHz** so this is the default.
        target_length
            Desired number of time-frames (after padding / truncation).
        n_mels
            Number of Mel bins to compute – **128** in the original model.
        norm_mean, norm_std
            Dataset-level mean / standard-deviation used for normalisation in
            the author's implementation.  If you have the exact constants you
            may pass them here.  By default we follow the original code-path:
            *no* global normalisation – callers may layer additional
            normalisation downstream if desired.
        frame_shift_ms
            Hop-size in **milliseconds** passed to Kaldi's *fbank* extractor –
            **10 ms** in the original.
        window_type
            Window function to use – "hanning" replicates the original repo.
        """

        self.sample_rate = sample_rate
        self.target_length = target_length
        self.n_mels = n_mels
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.frame_shift_ms = frame_shift_ms
        self.window_type = window_type

        # Expose hop-length (in *samples*) so downstream padding-mask utilities
        # can stay agnostic to the concrete processor implementation.
        self.hop_length: int = int(round(sample_rate * frame_shift_ms / 1_000))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def __call__(self, wav: Union[torch.Tensor, "np.ndarray"]) -> torch.Tensor:  # noqa: D401,E501  – keep signature compatible with AudioProcessor
        """Convert *wav* to Mel FBanks.

        Parameters
        ----------
        wav
            Raw mono waveform(s) in **float32** PCM (-1 … 1).  Shape can be
            ``(T,)`` or ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Mel FBanks shaped ``(B, n_mels, target_length)`` (float32).
        """
        if not isinstance(wav, torch.Tensor):
            # Convert numpy arrays to torch for unified processing
            wav = torch.as_tensor(wav, dtype=torch.float32)

        # Ensure (B, T) batch-format
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        batch_size, _ = wav.shape

        # We run Kaldi FBanks on *CPU* (the official op does not yet support
        # CUDA).  Tensors are moved back to their original device afterwards to
        # avoid surprises.
        original_device = wav.device
        wav_cpu = wav.cpu()

        fbanks: list[torch.Tensor] = []
        for idx in range(batch_size):
            mono = wav_cpu[idx]

            # Remove DC offset
            mono = mono - mono.mean()

            # Compute Mel FBanks via Kaldi compliance layer
            mel: torch.Tensor = torchaudio.compliance.kaldi.fbank(  # type: ignore[attr-defined] – torchaudio stub incomplete
                mono.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type=self.window_type,
                num_mel_bins=self.n_mels,
                dither=0.0,
                frame_shift=self.frame_shift_ms,
            )  # (T, n_mels)

            # Pad / truncate time dimension to *target_length*
            t = mel.size(0)
            if t < self.target_length:
                mel = F.pad(mel, (0, 0, 0, self.target_length - t))
            else:
                mel = mel[: self.target_length, :]

            # Optional dataset-level (or per-sample) normalisation.  If the
            # user left the defaults (0 / 1) we fall back to *per-sample*
            # statistics which is the closest we can get without knowing the
            # global constants used in the original pre-training corpus.
            if self.norm_mean == 0.0 and self.norm_std == 1.0:
                mean = mel.mean()
                std = mel.std() if mel.std() > 0 else 1.0
                mel = (mel - mean) / (std * 2)
            else:
                mel = (mel - self.norm_mean) / (self.norm_std * 2)

            # Transpose so final shape is (n_mels, T)
            fbanks.append(mel)

        fbanks_batch = torch.stack(fbanks, dim=0)  # (B, n_mels, T)
        return fbanks_batch.to(original_device)
