from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from representation_learning.configs import AudioConfig
from representation_learning.models.base_model import ModelBase


class Model(ModelBase):
    def __init__(
        self,
        num_classes: int = 1000,
        pretrained: bool = True,
        device: str = "cuda",
        audio_config: Optional[AudioConfig] = None,
        return_features_only: bool = False,
        model_id: str = "DBD-research-group/Bird-MAE-huge",
    ) -> None:
        # Call parent initializer with audio config
        super().__init__(device=device, audio_config=audio_config)

        # Store configuration
        self.return_features_only = return_features_only
        self.gradient_checkpointing = False
        self.model_id = model_id
        self.target_sample_rate = 32000  # BirdMAE expects 32kHz audio

        # Import transformers here to avoid import errors if not installed
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as e:
            raise ImportError(
                "transformers library is required for BirdMAE. "
                "Install with: pip install transformers"
            ) from e

        # Load the model and feature extractor
        if pretrained:
            self.model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        else:
            raise ValueError("BirdMAE currently only supports pretrained models")

        # Move model to device
        self.model = self.model.to(device)

        # Add classification head if needed
        if not self.return_features_only and num_classes != 1000:
            # Get the embedding dimension from the model
            # We'll determine this dynamically during first forward pass
            self.classifier = None
            self._embedding_dim = None

    def _ensure_classifier(self, embedding_dim: int, num_classes: int) -> None:
        """Create classifier head if it doesn't exist yet."""
        if self.classifier is None and not self.return_features_only:
            self.classifier = nn.Linear(embedding_dim, num_classes).to(self.device)
            self._embedding_dim = embedding_dim

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio input using BirdMAE's feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor of shape (batch_size, time_steps)

        Returns
        -------
        torch.Tensor
            Processed mel spectrogram tensor ready for BirdMAE
        """
        batch_size = x.shape[0]
        processed_batch = []

        # Process each sample in the batch
        for i in range(batch_size):
            audio_sample = x[i].cpu().numpy()

            # Use the feature extractor to get mel spectrogram
            # The feature extractor expects numpy array
            # Note: BirdMAE feature extractor may not accept sampling_rate parameter
            try:
                mel_spectrogram = self.feature_extractor(
                    audio_sample,
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt",
                )
            except TypeError:
                # Fallback: call without sampling_rate if not supported
                mel_spectrogram = self.feature_extractor(
                    audio_sample, return_tensors="pt"
                )

            # Extract the actual tensor from the feature extractor output
            if isinstance(mel_spectrogram, dict):
                # Feature extractor might return a dict with 'input_values' or similar
                for key in ["input_values", "input_features", "pixel_values"]:
                    if key in mel_spectrogram:
                        mel_tensor = mel_spectrogram[key]
                        break
                else:
                    # If none of the expected keys found, take the first tensor value
                    mel_tensor = next(iter(mel_spectrogram.values()))
            else:
                mel_tensor = mel_spectrogram

            # Ensure correct shape - BirdMAE expects [batch, 1, height, width]
            if mel_tensor.dim() == 3:  # [batch, height, width]
                mel_tensor = mel_tensor.unsqueeze(
                    1
                )  # Add channel dim -> [batch, 1, height, width]
            elif mel_tensor.dim() == 4 and mel_tensor.shape[1] != 1:
                # If we have multiple channels, take the first one or convert
                if mel_tensor.shape[1] > 1:
                    mel_tensor = mel_tensor[:, 0:1, :, :]  # Keep only first channel

            processed_batch.append(mel_tensor.squeeze(0))  # Remove batch dim

        # Stack all processed samples
        result = torch.stack(processed_batch, dim=0)
        return result.to(self.device)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for BirdMAE."""
        self.gradient_checkpointing = True
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (not used by BirdMAE)

        Returns
        -------
        torch.Tensor
            Model output (logits or features based on init flag)

        Raises
        ------
        ValueError
            If model output format is unexpected
        """
        # Process audio through feature extractor
        processed_input = self.process_audio(x)

        # Forward pass through BirdMAE
        outputs = self.model(processed_input)

        # Extract embeddings from model output
        if hasattr(outputs, "last_hidden_state"):
            # BirdMAE outputs shape [batch_size, embedding_dim] directly
            embeddings = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            # Direct tensor output
            embeddings = outputs
            if embeddings.dim() > 2:
                embeddings = embeddings.mean(dim=1)  # Pool if needed
        else:
            # For BirdMAE, the output might be a custom object
            # Try to access common attributes
            if hasattr(outputs, "prediction"):
                embeddings = outputs.prediction
            elif hasattr(outputs, "logits"):
                embeddings = outputs.logits
            elif hasattr(outputs, "hidden_states"):
                # Use the last hidden state
                hidden_states = outputs.hidden_states
                if isinstance(hidden_states, (list, tuple)):
                    embeddings = hidden_states[-1].mean(dim=1)  # Pool last layer
                else:
                    embeddings = hidden_states.mean(dim=1)
            else:
                # Debug: print available attributes
                attrs = [attr for attr in dir(outputs) if not attr.startswith("_")]
                raise ValueError(
                    f"Unexpected output format from BirdMAE. "
                    f"Type: {type(outputs)}, Available attributes: {attrs}"
                )

        # Ensure embeddings are 2D [batch_size, embedding_dim]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension if missing
        elif embeddings.dim() > 2:
            # Flatten extra dimensions
            embeddings = embeddings.view(embeddings.size(0), -1)

        # Ensure classifier exists if needed
        if not self.return_features_only:
            if self.classifier is None:
                # Determine number of classes from forward call context
                # For now, we'll defer classification head creation
                return embeddings
            else:
                return self.classifier(embeddings)

        return embeddings

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        layers: List[str],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        average_over_time: bool = True,
        freeze_backbone: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from BirdMAE.

        Parameters
        ----------
        x : torch.Tensor | dict[str, torch.Tensor]
            Input audio tensor or dictionary containing 'raw_wav'
        layers : List[str]
            List of layer names (ignored for BirdMAE, uses final embeddings)
        padding_mask : Optional[torch.Tensor]
            Padding mask for the input (not used by BirdMAE)
        average_over_time : bool
            Whether to average over time dimension (handled automatically)
        freeze_backbone : bool
            Whether to freeze the backbone and use torch.no_grad()

        Returns
        -------
        torch.Tensor
            Model embeddings
        """
        # Extract raw audio if provided as dict
        if isinstance(x, dict):
            x = x["raw_wav"]

        # Set model to eval mode for embedding extraction
        was_training = self.model.training
        self.model.eval()

        try:
            # Process audio and get embeddings (conditionally use torch.no_grad based on
            # freeze_backbone)
            if freeze_backbone:
                with torch.no_grad():
                    embeddings = self.forward(x, padding_mask)
            else:
                # Keep gradients enabled for fine-tuning
                embeddings = self.forward(x, padding_mask)
            return embeddings
        finally:
            # Restore training mode
            if was_training:
                self.model.train()
