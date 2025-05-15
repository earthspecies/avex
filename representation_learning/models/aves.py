import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model
from representation_learning.models.base_model import ModelBase


class AvesEmbedding(nn.Module):
    def __init__(self, sr, large=False):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        if large:
            config = self.load_config("configs/birdaves_bioxlarge.config")
        else:
            config = self.load_config("representation_learning/models/aves-base-core.torchaudio.model_config.json")
        self.model = wav2vec2_model(**config, aux_num_out=None)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://storage.googleapis.com/esp-public-files/birdaves/birdaves-biox-base.torchaudio.pt",
            map_location=device,
        )
        self.model.load_state_dict(state_dict)
        self.model.feature_extractor.requires_grad_(True)

        # bundle = torchaudio.pipelines.WAV2VEC2_BASE
        # self.model = bundle.get_model()

        self.sr = sr

    def load_config(self, config_path):
        with open(config_path, "r") as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig, padding_mask):
        # extract_feature in the torchaudio version will output all 12 layers' output, -1 to select the final one
        # print("sig", sig)

        out = self.model.extract_features(sig.float())[0][-1]
        padding_mask = padding_mask.bool()
        atts = ~padding_mask
        atts = atts.unsqueeze(1).float()
        atts = F.max_pool1d(atts, kernel_size=320, stride=320)
        atts = atts > 0
        padding_mask = ~atts

        return out, padding_mask

    def freeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.model.feature_extractor.requires_grad_(False)

    def unfreeze(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        self.model.feature_extractor.requires_grad_(True)


# --------------------------------------------------------------------------- #
#  High-level wrapper compatible with training pipeline
# --------------------------------------------------------------------------- #


# pylint: disable=abstract-method
# We inherit ModelBase directly; it already extends nn.Module.
class Model(ModelBase):
    """Wrapper that adapts :class:`AvesEmbedding` to ``ModelBase`` interface.

    The class mirrors the contracts of other model wrappers (EfficientNetB0,
    ResNetModel, …) so it can be selected through ``get_model`` and used by
    the generic training loop.

    Parameters
    ----------
    num_classes : int
        Number of target classes.  Ignored when *return_features_only* is
        ``True``.
    pretrained : bool, default ``True``
        Currently always *True* – BirdAVES weights are downloaded on demand.
    device : str
        Device on which to run the model.
    audio_config : Optional[Dict[str, Any]]
        Audio-processing parameters.  Must specify the input *sample_rate*; if
        ``None`` we default to the standard 16 kHz used by BirdAVES.
    return_features_only : bool, default ``False``
        When ``True`` the forward pass returns a pooled embedding vector
        (dimension = encoder hidden size).  When ``False`` a learnable linear
        classifier is appended and logits are returned instead.
    large : bool, default ``False``
        Select the *X-Large* BirdAVES backbone.  The default *Base* variant is
        substantially lighter and quicker to fine-tune.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,  # kept for API parity (always True atm.)
        device: str = "cuda",
        audio_config: Optional[dict] = None,
        return_features_only: bool = False,
        large: bool = False,
    ) -> None:
        # Initialise parent to set up audio pre-processor, device, etc.
        super().__init__(device=device, audio_config=audio_config)

        # ------------------------------------------------------------------ #
        # Backbone
        # ------------------------------------------------------------------ #
        if audio_config is None:
            sr = 16000
        else:
            # Support both Pydantic AudioConfig objects and plain dicts
            sr = getattr(audio_config, "sample_rate", None)
            if sr is None and isinstance(audio_config, dict):
                sr = audio_config.get("sample_rate", 16000)
            sr = sr or 16000
        self.backbone = AvesEmbedding(sr=sr, large=large)
        self.backbone.to(device)

        # Hidden size is determined by the Wav2Vec2 config – pull from encoder.
        try:
            hidden_dim = self.backbone.model.encoder.embed_dim  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback: assume 768 which is true for *Base* config
            hidden_dim = 768

        self.return_features_only = return_features_only

        if not self.return_features_only:
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.classifier.to(device)

    # ------------------------------------------------------------------ #
    # Helper functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool sequence hiding padded positions (mask=True → keep)."""
        mask = mask.unsqueeze(-1).type_as(x)  # (B, T, 1)
        x = x * mask
        summed = x.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Ensure audio tensor is on the right device and (potentially) run any
        # audio-level preprocessing (here overridden to a no-op).
        x = self.process_audio(x)

        if padding_mask is None:
            padding_mask = torch.zeros_like(x, dtype=torch.bool)

        features, new_mask = self.backbone(x, padding_mask)

        # features: (B, T, C)
        # pooled = self._masked_mean(features, ~new_mask)  # new_mask True=pad → invert
        pooled = features
        if self.return_features_only:
            return pooled
        else:
            return self.classifier(pooled)

    # Alias to satisfy other parts of the codebase that expect encode_audio()
    def encode_audio(self, audio: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return *pooled* AVES embedding (no classifier)."""
        feats, _m = self.backbone(audio, padding_mask)
        return self._masked_mean(feats, ~_m)

    # ------------------------------------------------------------------ #
    # Override ModelBase's audio preprocessing (raw waveform expected)
    # ------------------------------------------------------------------ #

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return waveform unchanged (AVES consumes raw audio)."""
        target_device = next(self.parameters()).device
        return x.to(target_device)
