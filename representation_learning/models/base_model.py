"""Base model class for representation learning models.

This module provides the base class for all representation learning models,
including functionality for hook management, embedding extraction, and audio processing.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from representation_learning.data.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    """Base class for all representation learning models.

    Provides common functionality for hook management, embedding extraction,
    and audio processing that all model implementations can inherit from.
    """

    def __init__(
        self, device: str, audio_config: Optional[Dict[str, Any]] = None
    ) -> None:
        super(ModelBase, self).__init__()
        self.device = device

        # Initialize audio processor if config is provided
        self.audio_processor = AudioProcessor(audio_config) if audio_config else None

        # Initialize hook management
        self._hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._hook_outputs: Dict[str, torch.Tensor] = {}
        self._layer_names: List[str] = []
        self._hook_layers: List[str] = []  # Standardized property name for all models

    def _discover_linear_layers(self) -> None:
        """Discover and cache all linear layer names in the model.

        This method can be overridden by subclasses to discover additional
        layer types beyond just nn.Linear layers.
        """
        if len(self._layer_names) == 0:  # Only discover once
            self._layer_names = []
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    self._layer_names.append(name)
            logger.info(
                f"Discovered {len(self._layer_names)} linear layers: "
                f"{self._layer_names}"
            )

    def _create_hook_fn(self, layer_name: str) -> callable:
        """Create a hook function for a specific layer.

        Returns:
            callable: A hook function that captures layer outputs.
        """

        def hook_fn(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            # Capture the tensor without detaching so gradients can propagate
            logger.debug(f"Hook triggered for layer: {layer_name}")
            if isinstance(output, dict):  # TODO: hacky - model-specific handling
                self._hook_outputs[layer_name] = output["x"]
            elif isinstance(output, tuple):
                self._hook_outputs[layer_name] = output[0]
            else:
                self._hook_outputs[layer_name] = output
            logger.debug(
                f"Captured output for {layer_name}: "
                f"{self._hook_outputs[layer_name].shape}"
            )

        return hook_fn

    def register_hooks_for_layers(self, layer_names: List[str]) -> List[str]:
        """Register forward hooks for the specified layers and return them.

        Parameters
        ----------
        layer_names : List[str]
            List of layer names to register hooks for.
            Special values:
            - 'all': Register hooks for all discoverable layers
            - 'last_layer': Register hooks for only the last (non-classification) layer

        Returns
        -------
        List[str]
            The resolved list of concrete layer names that hooks were
            registered for.

        Raises
        ------
        ValueError
            If a layer name is not found in the model.
        """
        # Discover layers if not already done
        if len(self._layer_names) == 0:
            self._discover_linear_layers()

        # Handle special cases
        if "all" in layer_names:
            # Replace 'all' with all discoverable layers, excluding the final
            # classification layer
            all_layers = self._layer_names.copy()
            # Remove 'all' from the list and add all discoverable layers
            layer_names = [name for name in layer_names if name != "all"]
            layer_names.extend(all_layers)
            # Remove duplicates while preserving order
            seen = set()
            unique_layers = []
            for name in layer_names:
                if name not in seen:
                    seen.add(name)
                    unique_layers.append(name)
            layer_names = unique_layers
            logger.info(f"Resolved 'all' to {len(layer_names)} layers: {layer_names}")

        if "last_layer" in layer_names:
            # Replace 'last_layer' with the last non-classification layer
            last_layer = self._get_last_non_classification_layer()

            if last_layer:
                # Replace 'last_layer' with the actual layer name
                layer_names = [
                    name if name != "last_layer" else last_layer for name in layer_names
                ]
                logger.info(
                    f"Resolved 'last_layer' to actual layer name: '{last_layer}'"
                )
            else:
                raise ValueError("No layers available for 'last_layer'")

        # Clear existing hooks
        self.deregister_all_hooks()

        # Store the target layers
        self._hook_layers = layer_names

        # Register hooks for each layer
        for layer_name in layer_names:
            try:
                module = self.get_submodule(layer_name)
                hook_handle = module.register_forward_hook(
                    self._create_hook_fn(layer_name)
                )
                self._hooks[layer_name] = hook_handle
            except AttributeError as err:
                raise ValueError(f"Layer '{layer_name}' not found in model") from err

        return layer_names

    def ensure_hooks_registered(self) -> None:
        """Ensure hooks are registered for previously requested layers.

        This is a resilience helper: if hooks were cleared by external lifecycle
        events (e.g., GC calling a probe's __del__), and we know which layers we
        should be hooked on (tracked in `_hook_layers`), re-register them.

        It is a no-op if hooks are already present or if no target layers were
        recorded.
        """
        if self._hooks:
            return
        if not self._hook_layers:
            return
        # Re-register hooks on the previously requested layers
        self.register_hooks_for_layers(self._hook_layers)

    def deregister_all_hooks(self) -> None:
        """Remove all registered hooks and clear outputs."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
        self._hook_outputs.clear()
        # Don't clear _hook_layers here as it's used to track which layers
        # should have hooks registered
        logger.debug("All hooks deregistered")

    def _get_last_non_classification_layer(self) -> Optional[str]:
        """Get the last non-classification layer name.

        Returns:
            Optional[str]: Name of the last non-classification layer, or None if
                not found
        """
        if not self._layer_names:
            return None

        # Look for the last layer that's not a classification head
        # Start from the end and work backwards
        for i in range(len(self._layer_names) - 1, -1, -1):
            name = self._layer_names[i]
            # Skip classification head layers (common patterns)
            # Check for explicit classification layer patterns
            if any(
                skip in name.lower()
                for skip in ["classifier", "head", "classifier_head"]
            ):
                continue
            # For models where all layers are internal (like BEATs),
            # return the actual last layer
            return name

        # If we can't find a non-classification layer, return the last layer
        # This handles cases where all layers are internal encoder layers
        return self._layer_names[-1]

    # def _get_last_non_classification_layer(self) -> Optional[str]:
    #     """Get the last non-classification layer name.

    #     Returns:
    #         Optional[str]: Name of the last non-classification layer, or None if
    #             not found
    #     """
    #     if not self._layer_names:
    #         return None

    #     # Look for the last layer that's not a classification head
    #     # Start from the second-to-last layer (assuming last is classifier)
    #     for i in range(len(self._layer_names) - 2, -1, -1):
    #         name = self._layer_names[i]
    #         # Skip classification head layers (common patterns)
    #         if not any(
    #             skip in name.lower()
    #             for skip in ["classifier", "head", "fc", "linear"]
    #         ):
    #             return name

    #     # If we can't find a non-classification layer, return the second-to-last layer
    #     # (assuming the last layer is the classifier)
    #     if len(self._layer_names) >= 2:
    #         return self._layer_names[-2]

    #     # Fallback to the last layer if only one layer exists
    #     return self._layer_names[-1]

    def _get_all_linear_layers(self) -> List[str]:
        """Get all linear layer names including the classification layer.

        Returns:
            List[str]: List of linear layer names.
        """
        return self._layer_names

    def _clear_hook_outputs(self) -> None:
        """Clear cached hook outputs without removing hooks."""
        self._hook_outputs.clear()

    def _cleanup_hooks(self) -> None:
        """Remove all registered hooks."""
        self.deregister_all_hooks()

    def __del__(self) -> None:
        """Cleanup hooks when the model is destroyed."""
        try:
            self._cleanup_hooks()
        except Exception:
            pass  # Ignore errors during cleanup

    def prepare_inference(self) -> None:
        self.model.eval()
        self.model = self.model.to(self.device)

    def prepare_train(self) -> None:
        self.model.train()
        self.model = self.model.to(self.device)

    def process_audio(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio input using the configured audio processor.
        Subclasses can override this method to implement custom audio processing.

        Args:
            x: Input tensor of shape (batch_size, time_steps)

        Returns:
            Processed audio tensor

        Raises:
            ValueError: If input tensor is None
        """
        if x is None:
            raise ValueError("Input tensor cannot be None")

        if self.audio_processor is not None:
            x = self.audio_processor(x)

        target_device = next(self.parameters()).device
        return x.to(target_device)

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory optimization.

        Subclasses must implement this method to enable checkpointing
        for their specific architecture.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support gradient checkpointing. "
            f"Please implement the enable_gradient_checkpointing method."
        )

    def batch_inference(self, batched_samples: torch.Tensor) -> torch.Tensor:
        """
        Perform batch inference on input samples.

        Args:
            batched_samples: Input tensor of shape (batch_size, time_steps)

        Returns:
            Concatenated embeddings tensor
        """
        embeds: List[torch.Tensor] = []
        for batch in tqdm(
            batched_samples,
            desc=" processing batches",
            position=0,
            leave=False,
        ):
            # Process audio if needed
            batch = self.process_audio(batch)

            embedding = self.__call__(batch)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)
        return torch.cat(embeds, axis=0)

    def extract_embeddings(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        padding_mask: Optional[torch.Tensor] = None,
        aggregation: str = "none",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract embeddings from all registered hooks in the model.

        Args:
            x: Input tensor or dictionary containing 'raw_wav'
            padding_mask: Optional padding mask for the input
            aggregation: Aggregation method for multiple layers ('mean', 'max',
                'cls_token', 'none')

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Model embeddings (tensor if
                aggregation!="none", list if False)

        Raises:
            ValueError: If no hooks are registered or no outputs are captured
        """
        # Clear previous hook outputs
        self._clear_hook_outputs()

        # Self-heal: re-register hooks if they were cleared externally
        self.ensure_hooks_registered()

        # Check if hooks are registered
        if not self._hooks:
            raise ValueError(
                "No hooks registered. Call register_hooks_for_layers() first."
            )

        try:
            # Forward pass (no torch.no_grad to allow fine-tuning when requested)
            if isinstance(x, dict):
                # Input provided as dictionary with explicit padding mask
                wav = x["raw_wav"]
                mask = x.get("padding_mask")
                expected_batch_size = wav.shape[0]
            else:
                # Input provided as tensor, no padding mask
                wav = x
                mask = padding_mask
                expected_batch_size = wav.shape[0]

            # Forward pass to trigger hooks
            self.forward(wav, mask)

            # Collect embeddings from hook outputs
            embeddings = []

            for layer_name in self._hook_outputs.keys():
                embeddings.append(self._hook_outputs[layer_name])
                logger.debug(
                    f"Found embedding for {layer_name}: "
                    f"{self._hook_outputs[layer_name].shape}"
                )

            logger.debug(f"Collected {len(embeddings)} embeddings")

            # Check if we got any embeddings
            if not embeddings:
                raise ValueError(
                    f"No layers found matching: {self._hook_outputs.keys()}"
                )

            # First, ensure all embeddings are in batch-first format
            for i in range(len(embeddings)):
                if embeddings[i].shape[0] != expected_batch_size:
                    # Transpose to batch-first format
                    embeddings[i] = embeddings[i].transpose(0, 1)

            # Process embeddings based on aggregation parameter
            if aggregation == "none":
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return embeddings
            else:
                # Average over time dimension if embeddings are 3D and concatenate
                for i in range(len(embeddings)):
                    if embeddings[i].dim() == 2:
                        # Already in correct shape
                        pass
                    elif embeddings[i].dim() == 3:
                        if aggregation == "mean":
                            embeddings[i] = embeddings[i].mean(dim=1)
                        elif aggregation == "max":
                            embeddings[i] = embeddings[i].max(dim=1)[
                                0
                            ]  # max returns (values, indices)
                        elif aggregation == "cls_token":
                            embeddings[i] = embeddings[i][:, 0, :]
                        else:
                            raise ValueError(
                                f"Unsupported aggregation method: {aggregation}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected embedding dimension: {embeddings[i].dim()}. "
                            f"Expected 2 or 3."
                        )

                # Concatenate all embeddings
                if len(embeddings) == 1:
                    return embeddings[0]
                else:
                    return torch.cat(embeddings, dim=1)

        finally:
            # Always clear hook outputs after extraction
            self._clear_hook_outputs()
