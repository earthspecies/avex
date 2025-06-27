from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from representation_learning.models.base_model import ModelBase


class FramewiseLinearProbe(nn.Module):
    """
    Lightweight *frame-wise* head for probing a frozen representation model.

    Instead of collapsing the time dimension to obtain one embedding per clip,
    this probe keeps the sequence dimension intact and applies a single
    `nn.Linear` to **every frame** (or timestep) independently.

    Parameters
    ----------
    base_model : Optional[ModelBase]
        Frozen backbone network.  Must be provided unless `feature_mode=True`.
    layers : List[str]
        Names of the layers whose outputs are concatenated to form the
        embeddings fed to the probe.
    num_classes : int
        Number of output classes for each timestep.
    device : str, default "cuda"
        Device to place both backbone and probe on.
    feature_mode : bool, default False
        If `True`, the probe expects *embeddings* as input rather than raw
        waveforms.  In that case `base_model` may be `None`.
    input_dim : Optional[int]
        Required only if `feature_mode=True` **and** `base_model is None`.
        Gives the per-frame embedding dimension.
    """

    def __init__(
        self,
        base_model: Optional[ModelBase],
        layers: List[str],
        num_classes: int,
        device: str = "cuda",
        feature_mode: bool = False,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.layers = layers
        self.feature_mode = feature_mode
        self.num_classes = num_classes

        # ---------------------------------------------------------------------
        # 1. Determine the size of **one frame’s** embedding
        # ---------------------------------------------------------------------
        if feature_mode:
            if input_dim is None and base_model is None:
                raise ValueError(
                    "When feature_mode=True and base_model is None, you must supply "
                    "`input_dim` so the probe knows the embedding size."
                )
            if input_dim is not None:
                frame_dim = input_dim
            else:
                # feature_mode=True but base_model *is* given → derive dim
                with torch.no_grad():
                    target_len = (
                        base_model.audio_processor.target_length_seconds
                        * base_model.audio_processor.sr
                    )
                    dummy = torch.randn(1, target_len, device=device)
                    emb = base_model.extract_embeddings(dummy, layers)
                    frame_dim = emb.shape[-1]  # last dim = per-frame size
        else:
            # We will compute embeddings inside `forward`
            if base_model is None:
                raise ValueError("base_model must be provided when feature_mode=False")
            with torch.no_grad():
                target_len = (
                    base_model.audio_processor.target_length_seconds
                    * base_model.audio_processor.sr
                )
                dummy = torch.randn(1, target_len, device=device)
                emb = base_model.extract_embeddings(
                    dummy, layers, average_over_time=False, framewise_embeddings=True
                )
                if isinstance(emb, list):
                    frame_dim = sum(e.shape[-1] for e in emb)
                else:
                    frame_dim = emb.shape[-1]

        # ---------------------------------------------------------------------
        # 2. A single Linear layer applied to every frame independently
        # ---------------------------------------------------------------------
        self.classifier = nn.Linear(frame_dim, num_classes).to(device)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            • If ``feature_mode=False`` – raw waveforms of shape (batch, time).
            • If ``feature_mode=True``  – embeddings of shape (batch, frames, dim).
        padding_mask : Optional[torch.Tensor], default None
            Boolean mask of shape (batch, frames) where ``True`` marks padded
            positions that should be ignored during loss computation.

        Returns
        -------
        torch.Tensor
            Frame-level logits of shape (batch, frames, num_classes)
        """
        # 1. Obtain embeddings with shape (B, T, D)
        if self.feature_mode:
            embeddings = x  # already (B, T, D)
        else:
            if self.base_model is None:
                raise ValueError("base_model must be provided when not in feature mode")
            embeddings = self.base_model.extract_embeddings(x, self.layers, average_over_time=False, framewise_embeddings=True)
            
            # Handle case where extract_embeddings returns a list of tensors
            if isinstance(embeddings, list):
                if len(embeddings) == 1:
                    embeddings = embeddings[0]  # Use first (and only) tensor
                else:
                    # Concatenate along feature dimension if multiple layers
                    embeddings = torch.cat(embeddings, dim=-1)

        # 2. Classify every frame independently
        #    nn.Linear works on the last dimension and is broadcast over the rest
        logits = self.classifier(embeddings)  # (B, T, num_classes)

        # 3. Optionally zero-out (or set to -inf) padded frames
        if padding_mask is not None:
            # shape -> (B, T, 1) for broadcasting
            logits = logits.masked_fill(padding_mask.unsqueeze(-1).bool(), 0.0)

        return logits
