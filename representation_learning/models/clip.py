import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from representation_learning.models.base_model import ModelBase
from representation_learning.models.efficientnet import (
    Model as EfficientNet,
)


class CLIPModel(ModelBase):
    """CLAP model combining EfficientNet for audio and RoBERTa for text."""

    def __init__(
        self,
        device: str,
        audio_config: Optional[Dict[str, Any]] = None,
        text_model_name: str = "roberta-base",
        projection_dim: int = 512,
        temperature: float = 0.07,
        efficientnet_variant: str = "b0",
        audio_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(device, audio_config)

        self.audio_encoder = EfficientNet(
            device=device,
            audio_config=audio_config,
            return_features_only=True,  # Get features before classifier
            efficientnet_variant=efficientnet_variant,
            pretrained=False,  # We'll load our own checkpoint if provided
        )

        # ------------------------------------------------------------------
        # Optionally load trained EfficientNet weights (ignore classifier)
        # ------------------------------------------------------------------
        if audio_checkpoint is not None:
            from pathlib import Path

            ckpt_path = Path(audio_checkpoint).expanduser()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Audio checkpoint not found: {ckpt_path}")

            state = torch.load(ckpt_path, map_location="cpu")
            # Detect common wrappers
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "model" in state and isinstance(state["model"], dict):
                    state = state["model"]

            from representation_learning.utils.utils import (
                sanitize_efficientnet_state_dict,
            )

            stripped_state = sanitize_efficientnet_state_dict(state)

            missing, unexpected = self.audio_encoder.model.load_state_dict(
                stripped_state, strict=False
            )
            if missing:
                print(
                    f"[CLIP] Warning: {len(missing)} EfficientNet weights "
                    f"missing after load"
                )
            if unexpected:
                print(
                    f"[CLIP] Warning: {len(unexpected)} unexpected keys "
                    f"when loading EfficientNet weights"
                )
            print(
                f"[CLIP] Loaded EfficientNet weights from {ckpt_path} "
                f"(classifier stripped)"
            )

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Projection heads: two-layer MLP (Linear → ReLU → Linear)
        # EfficientNet B0 has 1280 features, B1 has 1280 features too
        audio_feature_dim = 1280  # Both B0 and B1 have the same feature dimension
        hidden_dim = projection_dim
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        text_feature_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        init_value = torch.log(torch.tensor(1.0 / temperature))
        self.logit_scale = torch.nn.Parameter(init_value)

        # Move models to device
        self.audio_encoder.to(device)
        self.text_encoder.to(device)
        self.audio_projection.to(device)
        self.text_projection.to(device)

    # ------------------------------------------------------------------
    #  Convenience: expose audio encoder as `.backbone` so Trainer can
    #  automatically freeze/unfreeze it during two-stage fine-tuning.
    # ------------------------------------------------------------------

    @property  # type: ignore[override] – runtime-added attribute
    def backbone(self) -> nn.Module:  # noqa: D401 – simple property
        """Return the audio backbone (EfficientNet).

        Having this alias allows the generic Trainer `_activate_second_stage`
        logic to freeze/unfreeze CLIP's audio encoder just like other models.
        """

        return self.audio_encoder

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for both audio and text encoders."""
        # Enable checkpointing for audio encoder (EfficientNet)
        self.audio_encoder.enable_gradient_checkpointing()

        # Enable checkpointing for text encoder (HuggingFace transformer)
        self.text_encoder.gradient_checkpointing_enable()

    def encode_audio(
        self, audio: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode audio input using EfficientNet.

        Parameters
        ----------
        audio : torch.Tensor
            Audio input tensor

        Returns
        -------
        torch.Tensor
            Normalized audio embeddings
        """
        features = self.audio_encoder(audio, padding_mask)
        return F.normalize(self.audio_projection(features), dim=-1)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """Encode text input using RoBERTa.

        Parameters
        ----------
        text : list[str]
            List of text strings to encode

        Returns
        -------
        torch.Tensor
            Normalized text embeddings
        """
        current_device = next(self.parameters()).device
        tokens = self.text_tokenizer(
            text, padding=True, truncation=True, max_length=70, return_tensors="pt"
        ).to(current_device)

        outputs = self.text_encoder(**tokens)
        features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return F.normalize(self.text_projection(features), dim=-1)

    def forward(
        self, audio: torch.Tensor, text: list[str], padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing audio and text embeddings.

        Args:
            audio: Audio tensor of shape (batch_size, time_steps)
            text: List of text strings of length batch_size

        Returns:
            Tuple of (audio_embeddings, text_embeddings, logit_scale)
        """
        # Get normalized embeddings
        audio_embeddings = self.encode_audio(audio, padding_mask)
        text_embeddings = self.encode_text(text)

        LOGIT_SCALE_MAX = math.log(1.0 / 0.01)  # log(100)
        with torch.no_grad():
            self.logit_scale.clamp_(max=LOGIT_SCALE_MAX)

        return audio_embeddings, text_embeddings, self.logit_scale.exp()

    def extract_embeddings(
        self,
        x: torch.Tensor | Dict[str, torch.Tensor],
        layers: list[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
    ) -> torch.Tensor:
        """Extract audio embeddings from the CLIP model.

        Parameters
        ----------
        x : torch.Tensor | Dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav' and 'padding_mask'
        layers : list[str]
            List of layer names to extract embeddings from. Supported layers:
            - "audio_encoder" or "backbone": Raw audio features (1280-d)
            - "audio_projection": Projected audio features (projection_dim-d)
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input
        average_over_time : bool
            Kept for interface compatibility but ignored

        Returns
        -------
        torch.Tensor
            Audio embeddings from the specified layer
        """
        # Handle input format
        if isinstance(x, dict):
            raw_wav = x["raw_wav"]
            p_mask = x.get("padding_mask", padding_mask)
        else:
            raw_wav = x
            p_mask = padding_mask

        if p_mask is None:
            p_mask = torch.zeros(
                raw_wav.size(0),
                raw_wav.size(1),
                device=raw_wav.device,
                dtype=torch.bool,
            )

        # Extract audio features from EfficientNet
        audio_features = self.audio_encoder(raw_wav, p_mask)

        # Determine what to return based on requested layers
        if not layers:
            # Default behavior: return projected features
            return self.audio_projection(audio_features)

        # Check what layers are requested
        requested_layer = layers[0] if layers else "audio_projection"

        if requested_layer in ["audio_encoder", "backbone"]:
            # Return raw audio features (before projection)
            return audio_features
        elif requested_layer == "audio_projection":
            # Return projected features (after projection)
            return self.audio_projection(audio_features)
        else:
            # Fallback to base class behavior for other layer names
            # This allows using the hook-based extraction if needed
            return super().extract_embeddings(
                x,
                layers,
                padding_mask=padding_mask,
                average_over_time=average_over_time,
            )
