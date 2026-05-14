"""Plain ConvNeXt audio classification model.

ConvNextForImageClassification (1 audio channel, N-class head).

Architecture and preprocessing are fully configured via the YAML ``model_spec``
block.  When ``init_config`` or ``audio_config`` are omitted, defaults are read
from the packaged ``checkpoints/convnext_base.yml`` (same package as BEATs).

YAML quick-start::

    checkpoint_path: /path/to/convnext.ckpt
    model_spec:
      name: convnext
      pretrained: false
      init_config:
        depths: [3, 3, 27, 3]
        hidden_sizes: [128, 256, 512, 1024]
      audio_config:
        sample_rate: 32000
        n_fft: 2048
        hop_length: 256
        n_mels: 256
        extra_config:
          norm_mean: -13.369
          norm_std: 13.162

See ``avex/api/configs/checkpoints/convnext_base.yml`` for a ready-to-use
ConvNeXt-Base config that matches the BirdSet XCL training setup.
"""

import logging
from typing import Optional

import torch

from avex.models.base_model import ModelBase
from avex.models.utils.registry import load_packaged_yaml_mapping

logger = logging.getLogger(__name__)

_CONVNEXT_BASE_HF_ID = "facebook/convnext-base-224-22k"
_CHECKPOINT_CONFIGS_PKG = "avex.api.configs.checkpoints"
_CONVNEXT_BASE_YAML = "convnext_base"


def _load_convnext_base_yaml() -> dict:
    """Load and return the packaged convnext_base.yml mapping.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    ValueError
        If the packaged YAML is missing or malformed.
    """
    data = load_packaged_yaml_mapping(package=_CHECKPOINT_CONFIGS_PKG, name=_CONVNEXT_BASE_YAML)
    if not data:
        raise ValueError(
            f"Packaged YAML '{_CONVNEXT_BASE_YAML}.yml' not found in "
            f"'{_CHECKPOINT_CONFIGS_PKG}'.  Cannot resolve default architecture "
            "or mel parameters."
        )
    return data


def _parse_mel_params(audio_config: object) -> dict:
    """Return mel preprocessing parameters.

    If ``audio_config`` is None, parameters are loaded from the packaged
    ``convnext_base.yml``.  Otherwise ``audio_config`` values override the
    YAML defaults.

    Parameters
    ----------
    audio_config : AudioConfig | dict | None

    Returns
    -------
    dict
        Keys: ``n_fft``, ``hop_length``, ``n_mels``, ``sample_rate``,
        ``norm_mean``, ``norm_std``.
    """
    yaml_data = _load_convnext_base_yaml()
    yaml_audio = yaml_data.get("model_spec", {}).get("audio_config", {}) or {}
    yaml_extra = yaml_audio.get("extra_config", {}) or {}
    params = {
        "n_fft": yaml_audio["n_fft"],
        "hop_length": yaml_audio["hop_length"],
        "n_mels": yaml_audio["n_mels"],
        "sample_rate": yaml_audio["sample_rate"],
        "norm_mean": yaml_extra["norm_mean"],
        "norm_std": yaml_extra["norm_std"],
    }

    if audio_config is None:
        return params

    if hasattr(audio_config, "model_dump"):
        cfg = audio_config.model_dump()
    elif isinstance(audio_config, dict):
        cfg = audio_config
    else:
        return params

    for key in ("n_fft", "hop_length", "n_mels", "sample_rate"):
        if cfg.get(key) is not None:
            params[key] = cfg[key]

    extra = cfg.get("extra_config") or {}
    if "norm_mean" in extra:
        params["norm_mean"] = extra["norm_mean"]
    if "norm_std" in extra:
        params["norm_std"] = extra["norm_std"]

    return params


class Model(ModelBase):
    """Plain ConvNeXt audio classification model.

    Uses ``ConvNextForImageClassification`` (1 audio channel, N-class head).

    * ``pretrained=True`` — loads the ``facebook/convnext-base-224-22k`` ImageNet-22k
      backbone; the patch embedding is re-initialised for 1 channel.
    * ``pretrained=False`` — builds the architecture from ``init_config`` (or
      ``checkpoints/convnext_base.yml`` when omitted) with no network access;
      weights are loaded from a Lightning checkpoint via ``checkpoint_path``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: object = None,
        num_classes: Optional[int] = None,
        model_id: Optional[str] = None,
        init_config: Optional[dict] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        try:
            from transformers import ConvNextConfig, ConvNextForImageClassification
        except ImportError as e:
            raise ImportError("transformers is required.  Install with: pip install transformers") from e

        try:
            from torchaudio import transforms as T
        except ImportError as e:
            raise ImportError("torchaudio is required.  Install with: pip install torchaudio") from e

        if num_classes is None and not pretrained:
            raise ValueError(
                "num_classes is required when pretrained=False.  Pass num_classes= "
                "or let load_model() extract it from the checkpoint automatically."
            )

        self.gradient_checkpointing = False
        self.model_id = model_id or _CONVNEXT_BASE_HF_ID
        n_labels = num_classes or 1  # placeholder when pretrained=True and no head yet

        if pretrained:
            logger.info(
                "Loading ConvNeXt backbone from %s (1-channel patch embedding will be re-initialised from scratch)",
                _CONVNEXT_BASE_HF_ID,
            )
            self.model = ConvNextForImageClassification.from_pretrained(
                _CONVNEXT_BASE_HF_ID,
                num_labels=n_labels,
                num_channels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            if init_config:
                arch = init_config
            else:
                yaml_data = _load_convnext_base_yaml()
                arch = yaml_data.get("model_spec", {}).get("init_config")
                if not isinstance(arch, dict):
                    raise ValueError(
                        f"Missing/invalid 'init_config' in packaged YAML "
                        f"'{_CONVNEXT_BASE_YAML}.yml'.  Provide init_config explicitly."
                    )
            logger.info(
                "Building ConvNeXt from scratch: depths=%s hidden_sizes=%s",
                arch["depths"],
                arch["hidden_sizes"],
            )
            self.model = ConvNextForImageClassification(
                ConvNextConfig(
                    depths=arch["depths"],
                    hidden_sizes=arch["hidden_sizes"],
                    num_labels=n_labels,
                    num_channels=1,
                )
            )

        self.num_classes = num_classes

        mel = _parse_mel_params(audio_config)
        self._mel_params = mel
        self._spec_transform = T.Spectrogram(n_fft=mel["n_fft"], hop_length=mel["hop_length"], power=2.0)
        self._mel_scale = T.MelScale(n_mels=mel["n_mels"], sample_rate=mel["sample_rate"], n_stft=mel["n_fft"] // 2 + 1)
        self._db_scale = T.AmplitudeToDB(stype="power", top_db=80)

        self.model = self.model.to(device)

    def load_state_dict(self, state_dict: dict, strict: bool = False) -> object:
        """Load weights, transparently handling PyTorch-Lightning checkpoints.

        Unwraps the Lightning ``{"state_dict": {...}, "epoch": …}`` dict and
        strips the double ``model.model.`` / ``model._orig_mod.model.`` prefix
        added by Lightning wrappers.

        Returns
        -------
        object
            Result from the parent ``load_state_dict`` call.
        """
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            if isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]

        remapped = {}
        for k, v in state_dict.items():
            nk = k.replace("model._orig_mod.model.", "model.")
            nk = nk.replace("model.model.", "model.")
            remapped[nk] = v

        return super().load_state_dict(remapped, strict=strict)

    def _discover_linear_layers(self) -> None:
        """Discover the four ConvNeXt encoder stages for hook-based probing."""
        if self._layer_names:
            return

        self._layer_names = []
        for name, _ in self.named_modules():
            parts = name.split(".")
            if len(parts) == 5 and parts[:4] == ["model", "convnext", "encoder", "stages"] and parts[4].isdigit():
                self._layer_names.append(name)

        logger.info("Discovered %d ConvNeXt encoder stages: %s", len(self._layer_names), self._layer_names)

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to a normalised mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, time_steps)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, 1, n_mels, T)``, on ``self.device``.
        """
        mel = self._mel_params
        audio = x.cpu().float()
        spec = self._spec_transform(audio)
        mel_db = self._db_scale(self._mel_scale(spec))
        return ((mel_db - mel["norm_mean"]) / mel["norm_std"]).unsqueeze(1).to(self.device)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Extract embeddings from registered hooks.

        ConvNeXt encoder stages produce 4-D spatial feature maps ``(B, C, H, W)``.
        Matches the behaviour of :class:`~avex.models.efficientnet.Model`:

        * aggregation != ``"none"``: reduce over last dim (W/time), flatten
          remaining ``(B, C, H)`` → ``(B, C*H)``.
        * aggregation == ``"none"``: return raw tensors unchanged.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Embeddings from the hooked layers.

        Raises
        ------
        ValueError
            If no hooks are registered.
        """
        if not self._hooks:
            raise ValueError("No hooks registered. Call register_hooks_for_layers() first.")

        self._clear_hook_outputs()
        self.ensure_hooks_registered()

        wav = x["raw_wav"] if isinstance(x, dict) else x

        if freeze_backbone:
            with torch.no_grad():
                self.forward(wav, padding_mask)
        else:
            self.forward(wav, padding_mask)

        embeddings = list(self._hook_outputs.values())
        if not embeddings:
            raise ValueError("No outputs captured from registered hooks.")

        try:
            if aggregation == "none":
                return embeddings[0] if len(embeddings) == 1 else embeddings

            for i in range(len(embeddings)):
                if embeddings[i].dim() != 2:
                    if aggregation == "mean":
                        embeddings[i] = embeddings[i].mean(dim=-1)
                    elif aggregation == "max":
                        embeddings[i] = embeddings[i].max(dim=-1)[0]
                    elif aggregation == "cls_token":
                        embeddings[i] = embeddings[i][:, 0, :]
                    else:
                        raise ValueError(f"Unsupported aggregation method: {aggregation}")
                    if embeddings[i].dim() == 3:
                        embeddings[i] = embeddings[i].view(embeddings[i].shape[0], -1)

            return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=1)
        finally:
            self._clear_hook_outputs()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the ConvNeXt backbone."""
        self.gradient_checkpointing = True
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform, shape ``(batch_size, time_steps)``.
        padding_mask : Optional[torch.Tensor]
            Unused; kept for API compatibility.

        Returns
        -------
        torch.Tensor
            Classification logits, shape ``(batch_size, num_classes)``.
        """
        return self.model(self.process_audio(x)).logits
