"""AudioProtoPNet SED: frame-level sound event detection variant.

ConvNeXt-Base backbone + 2× bilinear add-on layers + spatial cosine prototype
head.  Differs from the BirdSet AudioProtoPNet in three key ways:

1. Uses ``ConvNextModel`` (backbone only, no classification head) so the
   output is a spatial feature map ``[B, C, H, W]``.
2. Prototype activation is spatial cosine similarity via conv2d with 4-D
   prototype vectors ``[P, C, 1, 1]``.
3. Two forward modes: clip-level (global H×W max-pool → logits) and
   frame-level (frequency max-pool only → sigmoid probabilities ``[B, T, C]``).

Checkpoint format — a local directory with three ``.pt`` files::

    checkpoint_dir/
        config.pt                # dict {"num_classes", "num_prototypes", "labels"}
        backbone_state_dict.pt   # ConvNextModel weights (may carry "backbone." prefix)
        head_state_dict.pt       # {"prototype_vectors", "last_layer.weight", "last_layer.bias"}

YAML quick-start::

    checkpoint_dir: ~/models/audioprotopnet-20/

    convnext_cfg:
      sample_rate: 32000
      n_fft: 2048
      hop_length: 256
      n_mels: 256
      norm_mean: -13.369
      norm_std: 13.162

    model_spec:
      name: audioprotopnet_sed
      pretrained: true
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from avex.models.base_model import ModelBase
from avex.models.convnext import _build_convnext_cfg, _load_convnext_base_yaml
from avex.utils import universal_torch_load

logger = logging.getLogger(__name__)

_CHECKPOINT_CONFIGS_PKG = "avex.api.configs.checkpoints"
_APN_SED_YAML = "audioprotopnet_sed"


# ── Helpers ───────────────────────────────────────────────────────────────


def _load_apn_sed_yaml() -> dict:
    from avex.models.utils.registry import load_packaged_yaml_mapping

    return load_packaged_yaml_mapping(package=_CHECKPOINT_CONFIGS_PKG, name=_APN_SED_YAML) or {}


def _build_backbone_config() -> object:
    """Build ConvNextConfig for the backbone from the packaged convnext_base.yml arch.

    Returns
    -------
    ConvNextConfig
        Architecture config for ConvNeXt-Base with 1 input channel.
    """
    from transformers import ConvNextConfig

    arch = _load_convnext_base_yaml().get("model_spec", {}).get("init_config", {})
    return ConvNextConfig(
        depths=arch["depths"],
        hidden_sizes=arch["hidden_sizes"],
        num_channels=1,
    )


def _get_sed_checkpoint_dir(checkpoint_dir: Optional[str]) -> str:
    """Return checkpoint directory from argument or packaged YAML top-level key.

    Returns
    -------
    str
        Resolved checkpoint directory path or GCS URI.

    Raises
    ------
    ValueError
        If no checkpoint_dir is provided and the packaged YAML has no default.
    """
    if checkpoint_dir is not None:
        return checkpoint_dir
    data = _load_apn_sed_yaml()
    cd = data.get("checkpoint_dir")
    if not (isinstance(cd, str) and cd):
        raise ValueError(
            f"No checkpoint_dir found in '{_APN_SED_YAML}.yml'. "
            "Set checkpoint_dir at the top level of the YAML or pass checkpoint_dir= explicitly."
        )
    return cd


def _cosine_activation(
    features: torch.Tensor,
    prototype_vectors: torch.Tensor,
    input_vector_length: int = 64,  # scale factor before normalisation; keeps activations in ~[0,1]
    n_eps_channels: int = 2,  # epsilon channels appended to prevent zero-length vectors
    epsilon_val: float = 1e-4,  # value of those epsilon channels
    # constants match AudioProtoNetClassificationHead.cos_activation in DBD-research-group/AudioProtoPNet
) -> torch.Tensor:
    """Spatial cosine similarity between feature maps and 4-D prototype vectors.

    Replicates ``AudioProtoNetClassificationHead.cos_activation`` from the
    original AudioProtoPNet HuggingFace implementation (inference path only).

    Parameters
    ----------
    features : torch.Tensor
        Shape ``[B, C, H, W]``.
    prototype_vectors : torch.Tensor
        Shape ``[num_prototypes, C, 1, 1]``.

    Returns
    -------
    torch.Tensor
        ReLU-gated cosine activations, shape ``[B, num_prototypes, H, W]``.
    """
    proto_h, proto_w = prototype_vectors.shape[2], prototype_vectors.shape[3]
    normalizing_factor = (proto_h * proto_w) ** 0.5

    eps_x = torch.full(
        (features.shape[0], n_eps_channels, features.shape[2], features.shape[3]),
        epsilon_val,
        device=features.device,
        dtype=features.dtype,
    )
    x = torch.cat((features, eps_x), dim=1)
    x_length = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + epsilon_val)
    x_normalized = (input_vector_length * x / x_length) / normalizing_factor

    eps_p = torch.full(
        (prototype_vectors.shape[0], n_eps_channels, proto_h, proto_w),
        epsilon_val,
        device=prototype_vectors.device,
        dtype=prototype_vectors.dtype,
    )
    appended_protos = torch.cat((prototype_vectors, eps_p), dim=1)
    proto_length = torch.sqrt(torch.sum(appended_protos**2, dim=1, keepdim=True) + epsilon_val)
    normalized_protos = appended_protos / (proto_length + epsilon_val) / normalizing_factor

    activations = F.conv2d(x_normalized, normalized_protos) / (input_vector_length * 1.01)
    return torch.relu(activations)


class _LinearLayerWithoutNegativeConnections(nn.Module):
    """Block-diagonal per-class linear layer with non-negative weights.

    Each class connects only to its own ``features_per_class`` prototype
    activations.  Non-negativity is enforced by ``relu`` in the forward pass
    (not by clamping), matching the original implementation.

    Parameters
    ----------
    in_features : int
        Total prototype activations (``num_classes × K``).
    out_features : int
        Number of classes.
    bias : bool
        Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features % out_features != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by out_features ({out_features})")
        self.in_features = in_features
        self.out_features = out_features
        self.features_per_output_class = in_features // out_features
        self.weight = nn.Parameter(torch.empty(out_features, self.features_per_output_class))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-class linear projection.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[..., in_features]``.  Leading dimensions are preserved.

        Returns
        -------
        torch.Tensor
            Shape ``[..., out_features]``.
        """
        leading_shape = x.shape[:-1]
        reshaped = x.view(*leading_shape, self.out_features, self.features_per_output_class)
        weight = torch.relu(self.weight)
        output = torch.einsum("...of,of->...o", reshaped, weight)
        if self.bias is not None:
            output = output + self.bias
        return output


# ── Main model ────────────────────────────────────────────────────────────


class Model(ModelBase):
    """AudioProtoPNet SED audio classification/detection model.

    ``forward(x)`` returns clip-level logits ``[B, num_classes]`` (global
    max-pool over H×W — same semantics as standard AudioProtoPNet).

    ``forward_frames(x)`` returns per-frame probabilities
    ``[B, T, num_classes]`` (max-pool over frequency only, sigmoid-activated —
    used for sound event detection metrics).

    Parameters
    ----------
    pretrained : bool
        Load weights from ``checkpoint_dir``.  ``False`` skips weight loading
        (random-initialised architecture, useful for testing).
    device : str
        Compute device.
    audio_config : object | None
        AudioConfig or dict overriding mel parameters from the packaged YAML.
    checkpoint_dir : str | None
        Directory with ``config.pt``, ``backbone_state_dict.pt``,
        ``head_state_dict.pt``.  Falls back to ``checkpoint_dir`` in
        ``audioprotopnet_sed.yml`` when ``None``.
    num_classes : int | None
        Required when ``pretrained=False``.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: object = None,
        checkpoint_dir: Optional[str] = None,
        convnext_cfg: Optional[dict] = None,
        num_classes: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(device=device, audio_config=audio_config)

        try:
            from torchaudio import transforms as T
        except ImportError as e:
            raise ImportError("torchaudio is required.  Install with: pip install torchaudio") from e

        try:
            from transformers import ConvNextModel as HFConvNextModel
        except ImportError as e:
            raise ImportError("transformers is required.  Install with: pip install transformers") from e

        mel = _build_convnext_cfg(convnext_cfg, yaml_data=_load_apn_sed_yaml())
        self._mel_params = mel
        self._spec_transform = T.Spectrogram(n_fft=mel.n_fft, hop_length=mel.hop_length, power=2.0).to(device)
        self._mel_scale = T.MelScale(n_mels=mel.n_mels, sample_rate=mel.sample_rate, n_stft=mel.n_fft // 2 + 1).to(
            device
        )
        self._db_scale = T.AmplitudeToDB(stype="power", top_db=80).to(device)

        if pretrained:
            resolved = _get_sed_checkpoint_dir(checkpoint_dir)
            logger.info("Loading AudioProtoPNet SED weights from %s", resolved)

            base = resolved.rstrip("/")
            metadata = universal_torch_load(f"{base}/config.pt", cache_mode="use", weights_only=False)
            n_classes: int = metadata["num_classes"]
            n_protos: int = metadata["num_prototypes"]
            ebird_labels: list[str] = metadata["labels"]

            backbone_cfg = _build_backbone_config()
            self.backbone = HFConvNextModel(backbone_cfg)
            backbone_sd = universal_torch_load(f"{base}/backbone_state_dict.pt", cache_mode="use", weights_only=True)
            backbone_sd = {k.removeprefix("backbone."): v for k, v in backbone_sd.items()}
            result = self.backbone.load_state_dict(backbone_sd, strict=False)
            if result.unexpected_keys:
                logger.warning("Unexpected keys in backbone checkpoint: %s", result.unexpected_keys)
            if result.missing_keys:
                logger.warning("Missing keys in backbone checkpoint: %s", result.missing_keys)

            self.add_on_layers: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear")

            head_sd = universal_torch_load(f"{base}/head_state_dict.pt", cache_mode="use", weights_only=True)
            self.prototype_vectors = nn.Parameter(head_sd["prototype_vectors"].to(device))

            protos_per_class = n_protos // n_classes
            self.last_layer = _LinearLayerWithoutNegativeConnections(
                in_features=n_protos, out_features=n_classes, bias=True
            )
            self.last_layer.weight.data.copy_(head_sd["last_layer.weight"])
            self.last_layer.bias.data.copy_(head_sd["last_layer.bias"])

            self.num_classes: int = n_classes
            self.ebird_codes: dict[int, str] = {i: c for i, c in enumerate(ebird_labels)}
            self.label_mapping: Optional[dict[int, str]] = self._build_label_mapping(self.ebird_codes)

            logger.info(
                "Loaded: %d classes, %d prototypes (%d/class)",
                n_classes,
                n_protos,
                protos_per_class,
            )

        else:
            if num_classes is None:
                raise ValueError("num_classes is required when pretrained=False for AudioProtoPNetSED.")
            backbone_cfg = _build_backbone_config()
            self.backbone = HFConvNextModel(backbone_cfg)
            self.add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
            n_protos = num_classes * 10
            feat_dim = backbone_cfg.hidden_sizes[-1]
            self.prototype_vectors = nn.Parameter(torch.randn(n_protos, feat_dim, 1, 1, device=device))
            self.last_layer = _LinearLayerWithoutNegativeConnections(
                in_features=n_protos, out_features=num_classes, bias=True
            )
            self.num_classes = num_classes
            self.ebird_codes = {}
            self.label_mapping = None

        self.backbone = self.backbone.to(device)
        self.add_on_layers = self.add_on_layers.to(device)
        self.last_layer = self.last_layer.to(device)

    @staticmethod
    def _build_label_mapping(ebird_codes: dict[int, str]) -> Optional[dict[int, str]]:
        """Return ``{class_id: common_name}`` using the bundled eBird taxonomy.

        Returns
        -------
        dict[int, str] | None
            Mapping of class index to common name, or raw eBird codes on taxonomy failure.
        """
        from avex.data.ebird_taxonomy import load as load_taxonomy

        try:
            taxonomy = load_taxonomy()
        except Exception as exc:
            logger.warning("Could not load eBird taxonomy, using raw codes: %s", exc)
            return dict(ebird_codes)
        return {idx: taxonomy[code]["common_name"] if code in taxonomy else code for idx, code in ebird_codes.items()}

    def _discover_linear_layers(self) -> None:
        """Discover the four ConvNeXt encoder stages for hook-based probing.

        SED path: ``backbone.encoder.stages.{0-3}``.
        """
        if self._layer_names:
            return
        self._layer_names = []
        for name, _ in self.named_modules():
            parts = name.split(".")
            if len(parts) == 4 and parts[:3] == ["backbone", "encoder", "stages"] and parts[3].isdigit():
                self._layer_names.append(name)
        logger.info(
            "Discovered %d AudioProtoPNet SED encoder stages: %s",
            len(self._layer_names),
            self._layer_names,
        )

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to normalised mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, time_steps]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, T]`` on ``self.device``.
        """
        audio = x.to(self.device).float()
        spec = self._spec_transform(audio)
        mel_db = self._db_scale(self._mel_scale(spec))
        return ((mel_db - self._mel_params.norm_mean) / self._mel_params.norm_std).unsqueeze(1)

    def _backbone_features(self, mel: torch.Tensor) -> torch.Tensor:
        """Backbone + add-on layers → spatial feature map.

        Returns
        -------
        torch.Tensor
            Shape ``[B, C, 2H, 2W]``.
        """
        return self.add_on_layers(self.backbone(mel).last_hidden_state)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
        freeze_backbone: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from registered hooks.

        ConvNeXt encoder stages produce 4-D ``[B, C, H, W]`` feature maps.

        * ``aggregation != "none"``: reduce over last dim (W/time), flatten
          remaining ``[B, C, H]`` → ``[B, C*H]``.
        * ``aggregation == "none"``: return raw tensors unchanged.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]

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

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Clip-level forward pass (global max-pool over H×W).

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform ``[B, time_steps]``.
        padding_mask : Optional[torch.Tensor]
            Unused; kept for API compatibility.

        Returns
        -------
        torch.Tensor
            Logits ``[B, num_classes]``.
        """
        features = self._backbone_features(self.process_audio(x))
        activations = _cosine_activation(features, self.prototype_vectors)
        max_act = activations.view(activations.shape[0], activations.shape[1], -1).max(dim=-1).values
        return self.last_layer(max_act)

    def forward_frames(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Frame-level forward pass (max-pool over frequency only, sigmoid).

        Parameters
        ----------
        x : torch.Tensor
            Raw waveform ``[B, time_steps]``.
        padding_mask : Optional[torch.Tensor]
            Unused.

        Returns
        -------
        torch.Tensor
            Per-frame probabilities ``[B, T, num_classes]``.
        """
        features = self._backbone_features(self.process_audio(x))
        activations = _cosine_activation(features, self.prototype_vectors)  # [B, P, H', W']
        pooled = activations.max(dim=2).values  # [B, P, W'] — max over freq
        logits = self.last_layer(pooled.permute(0, 2, 1))  # [B, W', num_classes]
        return torch.sigmoid(logits)
