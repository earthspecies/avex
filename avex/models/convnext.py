"""ConvNeXt audio encoder with a mel-spectrogram front-end.

Wraps Hugging Face ``ConvNextModel`` (1 input channel) behind the AVEX
``ModelBase`` API. The front-end replicates the AudioProtoPNet / BirdSet
preprocessing (``Spectrogram -> MelScale -> AmplitudeToDB -> z-score``) so that
checkpoints trained with that pipeline, such as the BirdCODE ConvNeXt backbone,
reproduce their training-time features exactly.

Modes
-----
* ``num_classes=None`` (or ``return_features_only=True``): encoder only.
  ``forward`` returns the last ConvNeXt stage as a spatial feature map
  ``(batch, channels, n_mels / 32, frames / 32)``.
* ``num_classes=N``: a linear head on the layer-normed, globally pooled
  features. ``forward`` returns logits ``(batch, N)``.

Configuration goes through ``ModelSpec.init_config`` (see
:class:`ConvNextInitConfig` for the keys and their defaults, which match
ConvNeXt-Base and the BirdSet XCL mel setup)::

    model_spec:
      name: convnext
      pretrained: false
      init_config:
        depths: [3, 3, 27, 3]
        hidden_sizes: [128, 256, 512, 1024]
        sample_rate: 32000
        n_fft: 2048
        hop_length: 256
        n_mels: 256
        norm_mean: -13.369
        norm_std: 13.162

``pretrained=False`` (the default) builds the architecture offline and expects
weights from ``checkpoint_path``; ``pretrained=True`` loads the ImageNet-22k
``facebook/convnext-base-224-22k`` weights from the Hub (the 3-channel patch
embedding is re-initialised for 1 channel) as a starting point for training.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, List, Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torchaudio import transforms as T

from avex.configs import AudioConfig
from avex.models.base_model import ModelBase

logger = logging.getLogger(__name__)

_IMAGENET_CONVNEXT_BASE = "facebook/convnext-base-224-22k"


@dataclasses.dataclass(frozen=True)
class ConvNextInitConfig:
    """Architecture and mel front-end parameters for :class:`Model`.

    Defaults are ConvNeXt-Base and the BirdSet XCL / AudioProtoPNet mel setup
    (32 kHz, 2048-point FFT, hop 256, 256 mel bins, 80 dB dynamic range,
    z-scored with the dataset mean and standard deviation).
    """

    depths: tuple[int, ...] = (3, 3, 27, 3)
    hidden_sizes: tuple[int, ...] = (128, 256, 512, 1024)
    sample_rate: int = 32_000
    n_fft: int = 2048
    hop_length: int = 256
    n_mels: int = 256
    top_db: float = 80.0
    norm_mean: float = -13.369
    norm_std: float = 13.162
    pretrained_model_id: str = _IMAGENET_CONVNEXT_BASE

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "ConvNextInitConfig":
        """Build a config from a ``ModelSpec.init_config`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any] | None
            Keys matching the dataclass fields. ``None`` yields the defaults.

        Returns
        -------
        ConvNextInitConfig
            Validated configuration.

        Raises
        ------
        ValueError
            If ``mapping`` contains keys that are not config fields.
        """
        if not mapping:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"Unknown convnext init_config keys: {unknown}. Supported keys: {sorted(known)}")
        values = dict(mapping)
        for key in ("depths", "hidden_sizes"):
            if key in values:
                values[key] = tuple(int(v) for v in values[key])
        return cls(**values)


class _MelFrontEnd(nn.Module):
    """``Spectrogram -> MelScale -> AmplitudeToDB -> z-score``, as in AudioProtoPNet.

    Submodule names (``spectrogram``, ``mel_scale``) are chosen so that the
    torchaudio buffers appear as ``mel_transform.spectrogram.window`` and
    ``mel_transform.mel_scale.fb`` in the state dict, matching the BirdCODE
    checkpoint layout.
    """

    def __init__(self, cfg: ConvNextInitConfig) -> None:
        super().__init__()
        self.spectrogram = T.Spectrogram(n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=2.0)
        self.mel_scale = T.MelScale(n_mels=cfg.n_mels, sample_rate=cfg.sample_rate, n_stft=cfg.n_fft // 2 + 1)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.norm_mean = float(cfg.norm_mean)
        self.norm_std = float(cfg.norm_std)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a waveform batch to a normalised log-mel spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``(batch, samples)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1, n_mels, frames)``.

        Notes
        -----
        The channel axis is added *before* ``AmplitudeToDB``. torchaudio takes
        the ``top_db`` floor from the maximum over the last three dimensions, so
        on a ``(batch, n_mels, frames)`` tensor the floor would be shared across
        the batch and a clip's features would depend on the other clips in it.
        With ``(batch, 1, n_mels, frames)`` the floor is per clip. (The upstream
        BirdCODE / AudioProtoPNet code uses the batch-wide floor; the two agree
        exactly at batch size 1.)
        """
        mel = self.mel_scale(self.spectrogram(waveform)).unsqueeze(1)
        mel_db = self.amplitude_to_db(mel)
        return (mel_db - self.norm_mean) / self.norm_std


class Model(ModelBase):
    """ConvNeXt audio encoder / classifier (see module docstring)."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        pretrained: bool = False,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        init_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        try:
            from transformers import ConvNextConfig, ConvNextModel
        except ImportError as e:  # pragma: no cover - transformers is a core dependency
            raise ImportError("transformers is required for the convnext model: pip install transformers") from e

        cfg = ConvNextInitConfig.from_mapping(init_config)
        self.init_config = cfg

        if audio_config is not None and audio_config.sample_rate != cfg.sample_rate:
            logger.warning(
                "audio_config.sample_rate=%d differs from init_config.sample_rate=%d; the mel front-end "
                "uses the latter. Deliver audio at %d Hz.",
                audio_config.sample_rate,
                cfg.sample_rate,
                cfg.sample_rate,
            )

        if num_classes is None:
            if not return_features_only:
                logger.info("num_classes is None for ConvNeXt; falling back to return_features_only=True.")
            return_features_only = True
        self.return_features_only = return_features_only
        self.num_classes = None if return_features_only else num_classes
        self.gradient_checkpointing = False

        if pretrained:
            logger.info(
                "Loading ConvNeXt backbone from %s (patch embedding re-initialised for 1 channel)",
                cfg.pretrained_model_id,
            )
            self.backbone = ConvNextModel.from_pretrained(
                cfg.pretrained_model_id,
                num_channels=1,
                ignore_mismatched_sizes=True,
            )
            loaded = self.backbone.config
            if tuple(loaded.depths) != cfg.depths or tuple(loaded.hidden_sizes) != cfg.hidden_sizes:
                logger.warning(
                    "init_config depths/hidden_sizes %s/%s ignored; using %s/%s from %s",
                    cfg.depths,
                    cfg.hidden_sizes,
                    tuple(loaded.depths),
                    tuple(loaded.hidden_sizes),
                    cfg.pretrained_model_id,
                )
        else:
            logger.info("Building ConvNeXt from scratch: depths=%s hidden_sizes=%s", cfg.depths, cfg.hidden_sizes)
            self.backbone = ConvNextModel(
                ConvNextConfig(depths=list(cfg.depths), hidden_sizes=list(cfg.hidden_sizes), num_channels=1)
            )

        self.mel_transform = _MelFrontEnd(cfg)

        self.classifier: Optional[nn.Linear] = None
        if not self.return_features_only:
            self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], int(num_classes))

        self.to(device)

    # ------------------------------------------------------------------ #
    #  Checkpoints                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _remap_detector_state_dict(state_dict: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], bool]:
        """Translate sound-event-detection ``FrameDetector`` keys to this wrapper's names.

        BirdCODE checkpoints store the backbone as ``encoder.backbone.*``, the
        mel buffers as ``encoder.mel_transform.*``, and a prototype head as
        ``encoder.prototype_vectors`` + ``classifier.{weight,bias}``. The head is
        not representable here and is dropped. Keys that the generic loader has
        already prefixed with ``backbone.`` (``backbone.encoder.backbone.*``) are
        handled as well.

        Returns
        -------
        tuple[dict[str, torch.Tensor], bool]
            The remapped state dict and whether a prototype head was dropped.
        """
        out: dict[str, torch.Tensor] = {}
        had_prototype_head = False
        for key, value in state_dict.items():
            k = key.removeprefix("backbone.") if key.startswith(("backbone.encoder.", "backbone.classifier.")) else key
            if k.startswith("encoder.backbone."):
                out["backbone." + k[len("encoder.backbone.") :]] = value
            elif k.startswith("encoder.mel_transform."):
                out["mel_transform." + k[len("encoder.mel_transform.") :]] = value
            elif k == "encoder.prototype_vectors" or k.startswith("classifier."):
                had_prototype_head = True
            else:
                out[key] = value
        return out, had_prototype_head

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Load weights, accepting native or BirdCODE ``FrameDetector`` layouts.

        When the checkpoint carries a prototype head and this model was built
        with a linear classifier, the classifier cannot be initialised from it;
        the model is switched to feature mode with a warning rather than left
        with a random head.

        Returns
        -------
        object
            Result of ``torch.nn.Module.load_state_dict``.
        """
        if any(k.startswith(("encoder.", "backbone.encoder.backbone.")) for k in state_dict):
            state_dict, had_prototype_head = self._remap_detector_state_dict(state_dict)
            if had_prototype_head and self.classifier is not None:
                logger.warning(
                    "Checkpoint has an AudioProtoPNet prototype head, which cannot initialise the %d-class linear "
                    "classifier; dropping the classifier and switching to feature mode.",
                    self.classifier.out_features,
                )
                self.classifier = None
                self.num_classes = None
                self.return_features_only = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------ #
    #  Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _discover_embedding_layers(self) -> None:
        """Expose the four ConvNeXt encoder stages (``backbone.encoder.stages.{0..3}``).

        Each stage emits a spatial map ``(batch, channels, freq, time)``; the
        last stage is what ``target_layers=["last_layer"]`` resolves to.
        """
        if self._layer_names:
            return
        prefix = "backbone.encoder.stages."
        self._layer_names = [
            name for name, _ in self.named_modules() if name.startswith(prefix) and name[len(prefix) :].isdigit()
        ]
        logger.info("Discovered %d ConvNeXt encoder stages: %s", len(self._layer_names), self._layer_names)

    # ------------------------------------------------------------------ #
    #  Audio processing                                                   #
    # ------------------------------------------------------------------ #

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw waveforms to normalised log-mel spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, samples)`` or ``(samples,)``; any float dtype.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1, n_mels, frames)`` on the model device.
        """
        x = super().process_audio(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.mel_transform(x)

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint the ConvNeXt backbone during training to save memory."""
        self.gradient_checkpointing = True

    def _backbone_features(self, mel: torch.Tensor) -> torch.Tensor:
        """Run the ConvNeXt backbone.

        Returns
        -------
        torch.Tensor
            Last-stage feature map ``(batch, channels, freq, time)``.
        """

        def run(inp: torch.Tensor) -> torch.Tensor:
            return self.backbone(pixel_values=inp).last_hidden_state

        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(run, mel, use_reentrant=False)
        return run(mel)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the model on raw audio.

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform ``(batch, samples)``.
        padding_mask : torch.Tensor | None
            Ignored; accepted for API compatibility across architectures.

        Returns
        -------
        torch.Tensor
            ``(batch, channels, freq, time)`` feature map in feature mode, or
            ``(batch, num_classes)`` logits when a classifier head is attached.
        """
        features = self._backbone_features(self.process_audio(x))
        if self.return_features_only or self.classifier is None:
            return features
        pooled = self.backbone.layernorm(features.mean([-2, -1]))
        return self.classifier(pooled)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                         #
    # ------------------------------------------------------------------ #

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from the registered hook layers.

        Encoder stages produce ``(batch, channels, freq, time)`` maps. With
        ``aggregation="none"`` they are returned unchanged. Otherwise each map is
        reduced over time (``mean``/``max``), flattened to ``(batch, channels * freq)``
        and the hooked layers are concatenated along the feature axis.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Raw waveform ``(batch, samples)`` or a dict with ``"raw_wav"``.
        padding_mask : torch.Tensor | None
            Ignored.
        aggregation : str
            ``"none"``, ``"mean"`` or ``"max"``. ``"cls_token"`` is rejected:
            ConvNeXt has no class token, and indexing the spatial map would
            select a single channel whose width depends on clip length.
        freeze_backbone : bool
            Run the forward pass under ``torch.no_grad()``.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            A single tensor, or a list of per-layer tensors when
            ``aggregation="none"`` and several layers are hooked.

        Raises
        ------
        ValueError
            If no hooks are registered, no outputs were captured, or the
            aggregation method is unknown.
        """
        self._clear_hook_outputs()
        self.ensure_hooks_registered()
        if not self._hooks:
            raise ValueError("No hooks registered. Call register_hooks_for_layers() first.")

        if aggregation not in ("none", "mean", "max"):
            raise ValueError(
                f"Unsupported aggregation method for convnext: {aggregation!r}. Use 'none', 'mean' or 'max' "
                "(ConvNeXt has no class token)."
            )

        wav = x["raw_wav"] if isinstance(x, dict) else x
        try:
            if freeze_backbone:
                with torch.no_grad():
                    self.forward(wav, padding_mask)
            else:
                self.forward(wav, padding_mask)

            ordered = self._hook_layers if self._hook_layers else list(self._hook_outputs.keys())
            embeddings = [self._hook_outputs[name] for name in ordered if name in self._hook_outputs]
            if not embeddings:
                raise ValueError("No outputs captured from registered hooks.")

            if aggregation == "none":
                return embeddings[0] if len(embeddings) == 1 else embeddings

            for i, emb in enumerate(embeddings):
                if emb.dim() == 2:
                    continue
                if aggregation == "mean":
                    emb = emb.mean(dim=-1)
                elif aggregation == "max":
                    emb = emb.max(dim=-1)[0]
                else:
                    raise ValueError(f"Unsupported aggregation method: {aggregation}")
                embeddings[i] = emb.reshape(emb.shape[0], -1)

            return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=1)
        finally:
            self._clear_hook_outputs()
